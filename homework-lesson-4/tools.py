from typing import Callable, Optional, Dict, Any, List
from pathlib import Path
from urllib.parse import urlparse

from ddgs import DDGS
import trafilatura

from config import settings


_EVENT_SINK: Optional[Callable[[Dict[str, Any]], None]] = None


def set_event_sink(sink: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    global _EVENT_SINK
    _EVENT_SINK = sink


def _emit(event: str, **payload: Any) -> None:
    if _EVENT_SINK is None:
        return
    try:
        _EVENT_SINK({"event": event, **payload})
    except Exception:
        pass


def _truncate_text(text: str, max_length: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


# -------------------------
# Реальні Python-функції
# -------------------------

def read_url(url: str) -> str:
    """
    Read a web page and return its main textual content.
    """
    _emit("tool_start", tool="read_url", url=url)

    if not isinstance(url, str) or not url.strip():
        return "Error: URL must be a non-empty string."

    url = url.strip()

    if not _is_valid_url(url):
        return "Error: Invalid URL. Expected a full URL like https://example.com/page"

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Error: Failed to download content from {url}"

        text = trafilatura.extract(downloaded, include_links=False, include_images=False)
        if not text:
            return f"Error: Could not extract readable text from {url}"

        text = text.strip()

        if len(text) > settings.max_url_content_length:
            text = text[:settings.max_url_content_length] + "\n\n[TRUNCATED]"

        _emit("tool_done", tool="read_url", url=url, chars=len(text))
        return text

    except Exception as e:
        return f"Error while reading URL {url}: {e}"


def web_search(query: str) -> List[Dict[str, str]]:
    """
    Search the web for relevant pages and return compact results.
    """
    _emit("tool_start", tool="web_search", query=query)

    query = (query or "").strip()
    if not query:
        _emit("tool_done", tool="web_search", query=query, results_count=0)
        return []

    try:
        raw_results = list(
            DDGS().text(query, max_results=settings.max_search_results) or []
        )
    except Exception as e:
        return [
            {
                "title": "Search error",
                "url": "",
                "snippet": _truncate_text(
                    f"DuckDuckGo search failed: {str(e)}",
                    settings.max_search_snippet_length,
                ),
            }
        ]

    results: List[Dict[str, str]] = []

    for item in raw_results[: settings.max_search_results]:
        if not isinstance(item, dict):
            continue

        title = _truncate_text(
            str(item.get("title") or ""),
            settings.max_search_title_length,
        )
        url = str(item.get("href") or "").strip()
        snippet = _truncate_text(
            str(item.get("body") or ""),
            settings.max_search_snippet_length,
        )

        if not (title or url or snippet):
            continue

        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
            }
        )

    _emit("tool_done", tool="web_search", query=query, results_count=len(results))
    return results


def write_report(filename: str, content: str) -> str:
    """
    Save the final Markdown report to a file.
    """
    if not isinstance(filename, str) or not filename.strip():
        return "Error: filename must be a non-empty string."

    if not isinstance(content, str) or not content.strip():
        return "Error: content must be a non-empty string."

    try:
        safe_name = Path(filename).name.strip()
        if not safe_name:
            return "Error: invalid filename."

        output_dir = Path(settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / safe_name
        _emit("tool_start", tool="write_report", filename=filename, chars=len(content))
        file_path.write_text(content, encoding="utf-8")
        _emit(
            "report_saved",
            tool="write_report",
            filename=filename,
            path=str(file_path.resolve()),
        )
        return f"Report saved successfully: {file_path.resolve()}"
    except Exception as e:
        return f"Error while saving report: {e}"


# -------------------------
# OpenAI Responses API tool schemas
# -------------------------
# Це саме те, що передається в client.responses.create(..., tools=OPENAI_TOOLS)

OPENAI_TOOLS = [
    {
        "type": "function",
        "name": "web_search",
        "description": (
            "Search the web for relevant pages on a topic and return a compact "
            "list of search results with title, url, and snippet. "
            "Use this when you need to discover sources before reading a page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "read_url",
        "description": (
            "Read a specific web page and return its main textual content. "
            "Use this after identifying a promising URL that needs summarization "
            "or detailed analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "A full HTTP or HTTPS URL, for example https://example.com/page"
                }
            },
            "required": ["url"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "write_report",
        "description": (
            "Save the final Markdown report to a file. "
            "Use this only when the report content is final and ready to be written to disk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The output filename, for example report.md"
                },
                "content": {
                    "type": "string",
                    "description": "The full Markdown content to save."
                }
            },
            "required": ["filename", "content"],
            "additionalProperties": False
        },
        "strict": True
    },
]


# -------------------------
# Registry для dispatch tool calls
# -------------------------

TOOL_REGISTRY = {
    "web_search": web_search,
    "read_url": read_url,
    "write_report": write_report,
}


def execute_tool_call(name: str, args: Dict[str, Any]) -> Any:
    tool_fn = TOOL_REGISTRY.get(name)
    if tool_fn is None:
        return f"Error: Unknown tool '{name}'"

    try:
        return tool_fn(**args)
    except TypeError as e:
        return f"Error: Invalid arguments for tool '{name}': {e}"
    except Exception as e:
        return f"Error while executing tool '{name}': {e}"
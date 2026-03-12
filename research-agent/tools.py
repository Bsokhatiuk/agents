from typing import List, Dict, Any

from ddgs import DDGS
from pydantic import BaseModel, Field
from langchain.tools import tool
from langgraph.config import get_stream_writer

from config import settings

from urllib.parse import urlparse

import trafilatura

from pathlib import Path


def _emit(event: str, **payload: Any) -> None:
    """Safe custom stream event emitter."""
    try:
        writer = get_stream_writer()
        writer({"event": event, **payload})
    except Exception:
        # Allows tools to be called outside LangGraph during local testing
        pass


def _is_valid_url(url: str) -> bool:
    """Перевіряє, що URL має коректну схему та хост."""
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False

def _is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


@tool
def read_url(url: str) -> str:
    """
    Read a web page and return its main textual content.

    This tool should be used when you need the actual content of a specific
    web page for summarization, fact extraction, comparison, or detailed analysis.
    It is especially useful after web_search, once you have identified a promising URL.

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

        _emit("tool_done", tool="read_url", url=url, chars=len(content))
        return text

    except Exception as e:
        return f"Error while reading URL {url}: {e}"

class WebSearchInput(BaseModel):
    query: str = Field(description="Search query")


@tool(args_schema=WebSearchInput)
def web_search(query: str) -> List[Dict[str, str]]:
    """
    Search the web for relevant pages on a topic and return a compact list of results.

    Use this tool when you need to discover sources, articles, documentation,
    blog posts, news pages, or other web pages related to a user query.
    This is the first step for open-ended internet research, especially when
    you do not yet know which specific page should be read.

    The tool returns search-result snippets, not full page content.
    """
    _emit("tool_start", tool="web_search", query=query)
    query = query.strip()
    if not query:
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
                "snippet": f"DuckDuckGo search failed: {str(e)}",
            }
        ]

    results: List[Dict[str, str]] = []

    for item in raw_results:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "").strip()
        url = str(item.get("href") or "").strip()
        snippet = str(item.get("body") or "").strip()

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

@tool
def write_report(filename: str, content: str) -> str:
    """
    Save the final Markdown report to a file.

    Use this tool when the final answer, summary, or report is ready and needs
    to be written to disk so it can be reviewed later. This tool should usually
    be called only once, at the end of the workflow, after the agent has finished
    gathering information and composing the final Markdown content.

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
        _emit("report_saved", tool="write_report", filename=filename, path=str(file_path.resolve()))
        return f"Report saved successfully: {file_path.resolve()}"
    except Exception as e:
        return f"Error while saving report: {e}"


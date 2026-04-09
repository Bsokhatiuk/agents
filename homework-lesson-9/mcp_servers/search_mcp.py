import json
import sys
from pathlib import Path
from urllib.parse import urlparse

# Add project root to path and load .env before any local imports
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import os
os.chdir(_ROOT)  # config uses relative paths for FAISS index

import trafilatura
from ddgs import DDGS
from fastmcp import FastMCP
from retriever import get_retriever


mcp_server = FastMCP(name="SearchMCP")

_retriever = get_retriever()

MANIFEST_PATH = _ROOT / "storage" / "faiss_index" / "manifest.json"

MAX_SEARCH_RESULTS = 5
MAX_URL_LENGTH = 5000
MAX_KNOWLEDGE_LENGTH = 1200


# ============================================================
# MCP Server: Define Tools
# ============================================================

@mcp_server.tool
def web_search(query: str) -> list[dict]:
    """Search the web and return a list of results with title, url, and snippet."""
    query = query.strip()
    if not query:
        return []
    try:
        raw = list(DDGS().text(query, max_results=MAX_SEARCH_RESULTS) or [])
    except Exception as e:
        return [{"title": "Search error", "url": "", "snippet": str(e)}]
    return [
        {
            "title": str(item.get("title") or ""),
            "url": str(item.get("href") or ""),
            "snippet": str(item.get("body") or ""),
        }
        for item in raw
        if isinstance(item, dict)
    ]


@mcp_server.tool
def read_url(url: str) -> str:
    """Fetch and extract readable text content from a web page."""
    url = url.strip()
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Error: invalid URL."
    except Exception:
        return "Error: invalid URL."
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Error: failed to download {url}"
        text = trafilatura.extract(downloaded, include_links=False, include_images=False) or ""
        if len(text) > MAX_URL_LENGTH:
            text = text[:MAX_URL_LENGTH] + "\n\n[TRUNCATED]"
        return text.strip() or f"No readable content at {url}"
    except Exception as e:
        return f"Error: {e}"


@mcp_server.tool
def knowledge_search(query: str) -> str:
    """Search the local knowledge base using hybrid retrieval and reranking."""
    docs = _retriever.invoke(query)
    if not docs:
        return "No relevant context found."
    blocks = []
    for i, doc in enumerate(docs[:5], start=1):
        blocks.append("\n".join([
            f"Document {i}",
            f"Source: {doc.metadata.get('source', 'unknown')}",
            f"Page: {doc.metadata.get('page', 'n/a')}",
            f"Chunk ID: {doc.metadata.get('chunk_id', 'unknown')}",
            f"Content: {doc.page_content[:MAX_KNOWLEDGE_LENGTH].strip()}",
        ]))
    return "\n\n".join(blocks)


# ============================================================
# MCP Server: Define Resources (read-only data)
# ============================================================

@mcp_server.resource("resource://knowledge-base-stats")
def get_kb_stats() -> str:
    """Return knowledge base stats: document count, total chunks, and last updated date."""
    if not MANIFEST_PATH.exists():
        return json.dumps({"error": "Manifest not found."})
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    total_chunks = sum(len(v.get("chunk_ids", [])) for v in files.values())
    return json.dumps({
        "documents": len(files),
        "total_chunks": total_chunks,
        "updated_at": manifest.get("updated_at"),
        "created_at": manifest.get("created_at"),
    })

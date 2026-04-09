import json
from pathlib import Path

from fastmcp import FastMCP
    


mcp_server = FastMCP(name="ReportMCP")

OUTPUT_DIR = Path(__file__).parent.parent / "output"


# ============================================================
# MCP Server: Define Resources (read-only data)
# ============================================================

@mcp_server.resource("resource://output-dir")
def get_output_dir() -> str:
    """Return the output directory path and a list of saved reports."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    files = [f.name for f in sorted(OUTPUT_DIR.iterdir()) if f.is_file()]
    return json.dumps({"path": str(OUTPUT_DIR), "reports": files})


@mcp_server.resource("resource://output-dir/{filename}")
def get_report(filename: str) -> str:
    """Read a saved report file by its name."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        available = [f.name for f in OUTPUT_DIR.iterdir() if f.is_file()]
        return json.dumps({"error": f"File '{filename}' not found", "available": available})
    return file_path.read_text(encoding="utf-8")


@mcp_server.tool
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

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        file_path = OUTPUT_DIR / safe_name
        file_path.write_text(content, encoding="utf-8")
        return f"Report saved successfully: {file_path.resolve()}"
    except Exception as e:
        return f"Error while saving report: {e}"

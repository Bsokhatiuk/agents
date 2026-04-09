import asyncio
import httpx
from fastmcp import Client as MCPClient
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain.agents import create_agent
from config import settings, SYSTEM_PROMPT

model = init_chat_model(settings.model_name, api_key=settings.api_key.get_secret_value())

ACP_BASE = settings.acp_base_url
REPORT_MCP_URL = settings.report_mcp_url


def _call_acp_agent(agent_name: str, content: str, timeout: int = 180) -> str:
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{ACP_BASE}/runs",
            json={
                "agent_name": agent_name,
                "input": [{"role": "user", "parts": [{"content": content, "content_type": "text/plain"}]}],
                "mode": "sync",
            },
        )
        response.raise_for_status()
        run = response.json()
        if run["status"] == "failed":
            raise RuntimeError(f"ACP agent '{agent_name}' failed: {run.get('error')}")
        return run["output"][-1]["parts"][0]["content"]


@tool
def plan(query: str) -> str:
    """Create a structured research plan for a query."""
    return _call_acp_agent("planner", query)


@tool
def research(query: str) -> str:
    """Research a topic using local knowledge and web sources, then return a structured Markdown report."""
    return _call_acp_agent("researcher", query)


@tool
def critique(query: str) -> str:
    """Review a research result and identify gaps, outdated claims, or structural issues."""
    return _call_acp_agent("critic", query)


async def _write_report_via_mcp(filename: str, content: str) -> str:
    async with MCPClient(REPORT_MCP_URL) as mcp_client:
        result = await mcp_client.call_tool("write_report", {"filename": filename, "content": content})
        return str(result)


@tool
def write_report(filename: str, content: str) -> str:
    """
    Save the final Markdown report to a file.

    Use this tool when the final answer, summary, or report is ready and needs
    to be written to disk so it can be reviewed later. This tool should usually
    be called only once, at the end of the workflow, after the agent has finished
    gathering information and composing the final Markdown content.
    """
    return asyncio.run(_write_report_via_mcp(filename, content))


supervisor = create_agent(
    model=model,
    tools=[plan, research, critique, write_report],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_report": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                }
            },
            description_prefix="Report save pending approval",
        ),
    ],
    checkpointer=InMemorySaver(),
)

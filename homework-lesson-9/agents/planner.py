import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import mcp_tools_to_langchain
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from config import settings, agent_planner_prompt
from schemas import ResearchPlan
from fastmcp import Client

model = init_chat_model(settings.model_name, api_key=settings.api_key.get_secret_value())


class MCPPlannerAgent:
    """Importable planner agent that uses only web_search and knowledge_search from MCP."""

    def __init__(self, mcp_url: str, allowed_tools: set[str]):
        self.mcp_url = mcp_url
        self.allowed_tools = allowed_tools

    async def ainvoke(self, inputs: dict) -> dict:
        async with Client(self.mcp_url) as mcp_client:
            mcp_tools = await mcp_client.list_tools()
            filtered_tools = [t for t in mcp_tools if t.name in self.allowed_tools]
            lc_tools = mcp_tools_to_langchain(filtered_tools, mcp_client)
            agent = create_agent(
                model=model,
                tools=lc_tools,
                system_prompt=agent_planner_prompt,
                response_format=ResearchPlan,
            )
            return await agent.ainvoke(inputs)


planner_agent = MCPPlannerAgent(settings.mcp_url, settings.planner_allowed_tools)
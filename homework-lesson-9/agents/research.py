import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from tools import mcp_tools_to_langchain
from langgraph.checkpoint.memory import MemorySaver
from config import settings, agent_research_prompt
from fastmcp import Client

model = init_chat_model(settings.model_name, api_key=settings.api_key.get_secret_value())


class MCPResearchAgent:
    """Importable research agent that uses web_search, read_url and knowledge_search from MCP."""

    def __init__(self, mcp_url: str, allowed_tools: set[str]):
        self.mcp_url = mcp_url
        self.allowed_tools = allowed_tools
        self.memory = MemorySaver()

    async def ainvoke(self, inputs: dict, config: dict = None) -> dict:
        async with Client(self.mcp_url) as mcp_client:
            mcp_tools = await mcp_client.list_tools()
            filtered_tools = [t for t in mcp_tools if t.name in self.allowed_tools]
            lc_tools = mcp_tools_to_langchain(filtered_tools, mcp_client)
            agent = create_agent(
                model=model,
                tools=lc_tools,
                system_prompt=agent_research_prompt,
                name="research",
                checkpointer=self.memory,
            )
            return await agent.ainvoke(inputs, config=config)


agent = MCPResearchAgent(settings.mcp_url, settings.research_allowed_tools)

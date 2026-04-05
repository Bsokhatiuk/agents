import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from tools import web_search, read_url, knowledge_search
from langgraph.checkpoint.memory import MemorySaver
from config import settings, agent_research_prompt

model = init_chat_model(settings.model_name, api_key=settings.api_key.get_secret_value())

tools = [web_search, read_url, knowledge_search]

memory = MemorySaver()

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=agent_research_prompt,
    name="research",
    checkpointer=memory,
)

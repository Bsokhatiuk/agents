from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from tools import write_report, web_search, read_url, knowledge_search
from langgraph.checkpoint.memory import MemorySaver
from config import settings, SYSTEM_PROMPT

model = init_chat_model(settings.model_name, api_key=settings.api_key.get_secret_value())

tools = [write_report, web_search, read_url, knowledge_search]

memory = MemorySaver()

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    name="my_agent",
    checkpointer=memory,
)

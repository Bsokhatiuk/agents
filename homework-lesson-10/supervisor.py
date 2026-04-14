from langchain.agents.middleware import HumanInTheLoopMiddleware
from typing import Literal
from agents import planner as pl_agetnt
from agents import research as res_agent 
from agents import critic as cr_agent   
from tools import write_report
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver 
from langchain.tools import tool
from langchain.agents import create_agent
from config import settings, SYSTEM_PROMPT

@tool
def research(query: str) -> str:
    """Research a topic using local knowledge and web sources, then return a structured Markdown report."""
    result = res_agent.agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

@tool
def plan(query: str) -> str:
    """Create a structured research plan for a query."""
    result = pl_agetnt.planner_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

@tool
def critique(query: str) -> str:
    """Review a research result and identify gaps, outdated claims, or structural issues."""
    result = cr_agent.critic_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


model = init_chat_model(settings.model_name, api_key=settings.api_key.get_secret_value())

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




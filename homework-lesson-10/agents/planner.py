import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from tools import web_search, knowledge_search
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from config import settings, agent_planner_prompt

model = init_chat_model(settings.model_name, api_key=settings.api_key.get_secret_value())

class ResearchPlan(BaseModel):
    goal: str = Field(description="What we are trying to answer")
    search_queries: list[str] = Field(description="Specific queries to execute")
    sources_to_check: list[str] = Field(description="'knowledge_base', 'web', or both")
    output_format: str = Field(description="What the final report should look like")

planner_agent = create_agent(
    model=model,
    tools=[web_search, knowledge_search],
    system_prompt=agent_planner_prompt,
    response_format=ResearchPlan,
)
# result["structured_response"] → validated ResearchPlan instance
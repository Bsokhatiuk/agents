import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Literal
from pydantic import BaseModel, Field
from tools import web_search, knowledge_search, read_url
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from config import settings, agent_critic_prompt


model = init_chat_model(settings.model_name, api_key=settings.api_key.get_secret_value())

class CritiqueResult(BaseModel):
    verdict: Literal["APPROVE", "REVISE"]
    is_fresh: bool = Field(description="Is the data up-to-date and based on recent sources?")
    is_complete: bool = Field(description="Does the research fully cover the user's original request?")
    is_well_structured: bool = Field(description="Are findings logically organized and ready for a report?")
    strengths: list[str] = Field(description="What is good about the research")
    gaps: list[str] = Field(description="What is missing, outdated, or poorly structured")
    revision_requests: list[str] = Field(description="Specific things to fix if verdict is REVISE")

critic_agent = create_agent(
    model=model,
    tools=[web_search, read_url, knowledge_search],
    system_prompt=agent_critic_prompt,
    response_format=CritiqueResult)
# ============================================================
# ACP Server: acp-sdk with two LangGraph agents
# ============================================================
import uuid
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Server
from agents import research, planner, critic

acp_server = Server()

print("ACP Server created (acp-sdk)")


@acp_server.agent(name="researcher", description="Research a topic using local knowledge and web sources, then return a structured Markdown report.")
async def researcher_handler(input: list[Message]) -> Message:
    user_text = input[-1].parts[0].content
    result = await research.agent.ainvoke(
        {"messages": [("user", user_text)]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )
    return Message(role="agent", parts=[MessagePart(content=result["messages"][-1].content)])


@acp_server.agent(name="planner", description="Create a structured research plan for a query.")
async def planner_handler(input: list[Message]) -> Message:
    user_text = input[-1].parts[0].content
    result = await planner.planner_agent.ainvoke({"messages": [("user", user_text)]})
    return Message(role="agent", parts=[MessagePart(content=result["structured_response"].model_dump_json())])


@acp_server.agent(name="critic", description="Review a research result and identify gaps, outdated claims, or structural issues.")
async def critic_handler(input: list[Message]) -> Message:
    user_text = input[-1].parts[0].content
    result = await critic.critic_agent.ainvoke({"messages": [("user", user_text)]})
    return Message(role="agent", parts=[MessagePart(content=result["structured_response"].model_dump_json())])


if __name__ == "__main__":
    acp_server.run(host="127.0.0.1", port=8903)
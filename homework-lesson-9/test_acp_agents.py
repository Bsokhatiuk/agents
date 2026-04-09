import asyncio
import httpx

ACP_URL = "http://127.0.0.1:8903"
QUERY = "What is a multi-agent system?"


async def call_agent(agent_name: str, query: str) -> str:
    payload = {
        "agent_name": agent_name,
        "input": [{"role": "user", "parts": [{"content": query, "content_type": "text/plain"}]}],
        "mode": "sync",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(f"{ACP_URL}/runs", json=payload)
        response.raise_for_status()
        run = response.json()
        return run["output"][-1]["parts"][0]["content"]


async def test_agent(agent_name: str, query: str) -> None:
    print(f"\n{'='*50}")
    print(f"Agent: {agent_name}")
    print(f"Query: {query}")
    print("=" * 50)
    output = await call_agent(agent_name, query)
    print(f"Response ({len(output)} chars):\n{output[:400]}{'...' if len(output) > 400 else ''}")


async def main():
    await test_agent("planner", QUERY)
    await test_agent("researcher", QUERY)
    await test_agent(
        "critic",
        f"Original request: {QUERY}\n\nFindings: Multi-agent systems consist of multiple interacting autonomous agents that collaborate to solve complex problems.",
    )


if __name__ == "__main__":
    asyncio.run(main())

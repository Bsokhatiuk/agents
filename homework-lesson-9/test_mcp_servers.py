import asyncio
from fastmcp import Client


async def inspect_server(name: str, url: str):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  {url}")
    print(f"{'='*50}")
    async with Client(url) as client:
        tools = await client.list_tools()
        print(f"\nTools ({len(tools)}):")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        resources = await client.list_resources()
        print(f"\nResources ({len(resources)}):")
        for r in resources:
            print(f"  - {r.uri}: {r.description}")


async def main():
    await inspect_server("SearchMCP", "http://127.0.0.1:8901/mcp")
    await inspect_server("ReportMCP", "http://127.0.0.1:8902/mcp")


if __name__ == "__main__":
    asyncio.run(main())

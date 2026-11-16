import asyncio
import sys

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_types import TextContent

USAGE = """Usage:
  python weather_cli.py tools
  python weather_cli.py alerts <STATE>            e.g. alerts CA
  python weather_cli.py forecast <LAT> <LON>      e.g. forecast 37.77 -122.42"""


def parse_call(argv: list[str]) -> tuple[str, dict] | None:
    """Map command line arguments to an MCP tool name and its arguments"""
    match argv:
        case ["alerts", state]:
            return "get_alerts", {"state": state.upper()}
        case ["forecast", lat, lon]:
            return "get_forecast", {"latitude": float(lat), "longitude": float(lon)}
    return None


async def main() -> None:
    argv = sys.argv[1:]
    call = parse_call(argv)
    print(f"Parsed call: {call}")  # Debugging line
    if call is None and argv != ["tools"]:
        print(USAGE)
        sys.exit(1)

    # Start the server with this same interpreter so it shares the Poetry environment
    params = StdioServerParameters(command=sys.executable, args=["weather.py"])

    async with Client(stdio_client(params)) as client:
        if call is None:
            tool_list = await client.list_tools()
            for tool in tool_list.tools:
                print(f"{tool.name}: {tool.description}")
            return

        name, arguments = call
        result = await client.call_tool(name, arguments)
        for block in result.content:
            if isinstance(block, TextContent):
                print(block.text)
        if result.is_error:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

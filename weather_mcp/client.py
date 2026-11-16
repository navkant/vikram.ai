import asyncio
import sys

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_types import TextContent

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

MODEL = "claude-opus-5"
# The model thinks adaptively unless told otherwise, and max_tokens caps thinking
# plus the reply, so leave room for both.
MAX_TOKENS = 10000
MAX_TOOL_TURNS = 10
anthropic = Anthropic()


def server_params(server_script_path: str) -> StdioServerParameters:
    """Describe the subprocess that runs an MCP server

    Args:
        server_script_path: Path to the server script (.py or .js)
    """
    if server_script_path.endswith(".py"):
        command = "python"
    elif server_script_path.endswith(".js"):
        command = "node"
    else:
        raise ValueError("Server script must be a .py or .js file")

    return StdioServerParameters(command=command, args=[server_script_path])


async def process_query(client: Client, query: str) -> str:
    """Process a query using Claude and available tools"""
    messages = [
        {
            "role": "user",
            "content": query
        }
    ]

    tool_list = await client.list_tools()
    available_tools = [{
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema
    } for tool in tool_list.tools]

    final_text = []

    # Initial Claude API call
    response = anthropic.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=messages,
        tools=available_tools
    )

    for _ in range(MAX_TOOL_TURNS):
        # Process response and collect tool calls
        tool_uses = []
        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_uses.append(content)

        if not tool_uses:
            return "\n".join(final_text)

        tool_results = []
        for tool_use in tool_uses:
            # Execute tool call
            result = await client.call_tool(tool_use.name, tool_use.input)
            final_text.append(f"[Calling tool {tool_use.name} with args {tool_use.input}]")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": "\n".join(
                    block.text
                    for block in result.content
                    if isinstance(block, TextContent)
                ),
                "is_error": result.is_error
            })

        # Replay the assistant turn as returned, then answer all of its
        # tool calls in a single user message
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        # Get next response from Claude
        response = anthropic.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
            tools=available_tools
        )

    # The turn cap was hit. Keep the last response's text. If it asked for
    # more tools, say they were not run.
    final_text.extend(content.text for content in response.content if content.type == 'text')
    if any(content.type == 'tool_use' for content in response.content):
        final_text.append(f"[Stopped after {MAX_TOOL_TURNS} tool-use turns]")
    return "\n".join(final_text)


async def chat_loop(client: Client) -> None:
    """Run an interactive chat loop"""
    print("\nMCP Client Started!")
    print("Type your queries or 'quit' to exit.")

    while True:
        try:
            query = (await asyncio.to_thread(input, "\nQuery: ")).strip()
        except EOFError:
            break

        if query.lower() == 'quit':
            break

        try:
            response = await process_query(client, query)
            print("\n" + response)
        except Exception as e:
            print(f"\nError: {e}")


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    async with Client(stdio_client(server_params(sys.argv[1]))) as client:
        tool_list = await client.list_tools()
        tool_names = [tool.name for tool in tool_list.tools]
        print("\nConnected to server with tools:", tool_names)

        await chat_loop(client)


if __name__ == "__main__":
    asyncio.run(main())
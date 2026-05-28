import asyncio
import os
import re
import shlex
import shutil
import subprocess
from typing import List
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.settings import settings


def build_server_params(token: str) -> StdioServerParameters:

    npx_path = shutil.which("npx")
    if not npx_path:
        raise RuntimeError(
            "Could not find 'npx'. Install Node.js inside your Linux environment, "
            "or set GITHUB_MCP_SERVER_COMMAND to a working GitHub MCP server executable."
        )

    node_version_result = subprocess.run(
        ["node", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    node_version = node_version_result.stdout.strip()
    match = re.match(r"v(\d+)", node_version)

    node_major_version = int(match.group(1))
    if node_major_version < 18:
        raise RuntimeError(
            f"Detected Node.js {node_version}, which is too old to run the GitHub MCP server reliably. "
            "Install a recent Node.js release inside WSL so both 'node' and 'npx' resolve to Linux binaries, "
            "or set GITHUB_MCP_SERVER_COMMAND to a working GitHub MCP server executable."
        )

    return StdioServerParameters(
        command='npx',
        args=["-y", "-p", "@modelcontextprotocol/server-github", "mcp-server-github"],
        env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": token},
    )


def get_all_github_tools() -> List[BaseTool]:
    # syncronous fetching
    return asyncio.run(fetch_tools())

def get_specific_github_tools(tool_names: List[str]) -> List[BaseTool]:
    all_tools = get_all_github_tools()
    tool_name_set = set(tool_names)
    selected_tools = [tool for tool in all_tools if tool.name in tool_name_set]
    missing_tools = tool_name_set - set(tool.name for tool in selected_tools)
    if missing_tools:
        raise ValueError(f"Requested GitHub tools not found: {', '.join(missing_tools)}")
    return selected_tools

def get_github_tools_names() -> List[str]:
    tools = get_all_github_tools()
    return [tool.name for tool in tools]

async def fetch_tools() -> List[BaseTool]:
    token = settings.GITHUB_PERSONAL_ACCESS_TOKEN

    if not token:
        raise ValueError("GITHUB_PERSONAL_ACCESS_TOKEN not found in environment")
    server_params = build_server_params(token)

    # Note: This opens a connection, gets the tool definitions, and returns them.
    # The adapter handles re-opening the connection when the tool is actually called.
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await load_mcp_tools(session)
        

if __name__ == "__main__":
    # get names
    tool_names = get_github_tools_names()
    print("Available GitHub Tools:")
    for name in tool_names:
        print(f"- {name}")

    

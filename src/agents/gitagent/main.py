"""Git Agent main module."""

from __future__ import annotations
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.tools.git_tooling.main import get_github_tools_names, get_specific_github_tools

acceptable_tool_names = ['create_or_update_file',
                        'create_repository',
                        'push_files', 
                        'list_commits'
                        ]

read_only_tool_names = ["search_repositories",
                        "get_file_contents",
                        "search_code",
                        "list_commits",
                        "list_issues",
                        "search_issues",
                        "get_issue",
                        "list_pull_requests",
                        "get_pull_request",
                        "get_pull_request_files",
                        "get_pull_request_status",
                        "get_pull_request_comments",
                        "get_pull_request_reviews",
                        "search_users",
                        "add_issue_comment"]

def build_git_agent():
    """Build a GitHub-focused agent that can use the available MCP Git tools."""
    from src.settings import settings
    print("1")

    if settings.OPEN_ROUTER_KEY is None:
        raise ValueError("OPEN_ROUTER_KEY must be set in the .env file.")
    print("2")

    model = ChatOpenAI(
        api_key=SecretStr(settings.OPEN_ROUTER_KEY),
        base_url="https://openrouter.ai/api/v1",
        model="gpt-4o-mini",
        temperature=0.1,
        max_completion_tokens=10000,
    )
    print("3")
    all_tool_names = get_github_tools_names()  # Call this function to ensure tools are registered and available
    print("3.5")
    available_tools = get_specific_github_tools(
        [name for name in read_only_tool_names if name in all_tool_names]
    )
    print("4")

    if not available_tools:
        raise ValueError("No acceptable GitHub tools are available. Please check the tool names and availability.")
    print("5")

    system_prompt = (
        "You are a GitHub automation assistant. Use the available GitHub tools to "
        "inspect repositories, create or update files, list commits, and perform other "
        "GitHub-related tasks requested by the user. Be precise and only use tools that are relevant."
        "You have access to the following tools: " + ", ".join(tool.name for tool in available_tools) + ". "
    )
    print("6")
    return create_agent(
        model=model,
        system_prompt=system_prompt,
        tools=available_tools,
    )

if __name__ == "__main__":
    agent = build_git_agent()
    

    # print all tool names in rows of 4
    tool_names = get_github_tools_names()
    print("Available tools:")
    for i in range(0, len(tool_names), 4):
        print("  |  ".join(tool_names[i:i+4]))
    print("Git agent built successfully:", agent)
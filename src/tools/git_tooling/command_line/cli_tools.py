# src/tools/git_tooling/cli_tools.py

import json
from pydantic import BaseModel
from langchain_core.tools import tool
from src.tools.git_tooling.command_line.helper import GitCommitSchema, GitDiffSchema, GitLogSchema, GitRepoSchema, _as_json

from src.tools.git_tooling.command_line.cli_handler import GitCLI, GitCLIError


def _git(repo_path: str) -> GitCLI:
    return GitCLI(repo_path)


@tool(args_schema=GitRepoSchema)
def git_status(repo_name: str) -> str:
    """Check the current Git status for a repository inside the workplace folder."""
    try:
        return _as_json(_git(repo_name).status())
    except GitCLIError as error:
        return _as_json({"success": False, "stderr": str(error)})


@tool(args_schema=GitDiffSchema)
def git_diff(repo_name: str, args: list[str] = []) -> str:
    """Show Git diff output. Optional args may include paths or flags such as --staged."""
    try:
        return _as_json(_git(repo_name).diff(*args))
    except GitCLIError as error:
        return _as_json({"success": False, "stderr": str(error)})


@tool(args_schema=GitLogSchema)
def git_log(repo_name: str, limit: int = 10, oneline: bool = True) -> str:
    """Read recent commit history for a repository."""
    try:
        return _as_json(_git(repo_name).log(limit=limit, oneline=oneline))
    except GitCLIError as error:
        return _as_json({"success": False, "stderr": str(error)})


@tool(args_schema=GitCommitSchema)
def git_stage_and_commit(repo_name: str, items: list[str], message: str) -> str:
    """Stage selected files and create a commit."""
    try:
        return _as_json(_git(repo_name).stage_and_commit_items(*items, message=message))
    except GitCLIError as error:
        return _as_json({"success": False, "stderr": str(error)})
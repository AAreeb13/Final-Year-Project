
# Schemas
from typing import TypedDict

from pydantic import BaseModel, json


class GitRepoSchema(BaseModel):
    repo_path: str


class GitDiffSchema(BaseModel):
    repo_path: str
    args: list[str] = []


class GitLogSchema(BaseModel):
    repo_path: str
    limit: int = 10
    oneline: bool = True


class GitCommitSchema(BaseModel):
    repo_path: str
    items: list[str]
    message: str

# Helper functions
def _as_json(result: dict) -> str:
    return json.dumps(result)


# for defining types and exceptions related to GitCLI

class GitCLIError(Exception):
    pass

class GitOutput(TypedDict):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    success: bool


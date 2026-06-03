from __future__ import annotations

import shlex
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


AgentStage = Literal["start", "coding", "execution"]
CompletionStatus = Literal["complete", "not complete"]


class FileTask(BaseModel):
    task_id: str
    relative_path: str
    action: Literal["create", "edit", "delete", "review"] = "create"
    description: str
    depends_on: list[str] = Field(default_factory=list)


class DependencyPlan(BaseModel):
    packages: list[str] = Field(default_factory=list)
    setup_commands: list[list[str]] = Field(default_factory=list)
    network_required: bool = False
    notes: str = ""

    @field_validator("setup_commands", mode="before")
    @classmethod
    def normalize_setup_commands(cls, value: Any) -> Any:
        return _normalize_commands(value)


class TestPlan(BaseModel):
    testing_framework: str = "pytest"
    test_commands: list[list[str]] = Field(default_factory=list)
    notes: str = ""

    @field_validator("test_commands", mode="before")
    @classmethod
    def normalize_test_commands(cls, value: Any) -> Any:
        return _normalize_commands(value)


class SupervisorOutput(BaseModel):
    project_summary: str = ""
    functional_requirements: list[str] = Field(default_factory=list)
    non_functional_requirements: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    architecture_design: list[str] = Field(default_factory=list)
    repository_structure: list[str] = Field(default_factory=list)
    file_tasks: list[FileTask] = Field(default_factory=list)
    dependency_plan: DependencyPlan = Field(default_factory=DependencyPlan)
    test_plan: TestPlan = Field(default_factory=TestPlan)


class AgentProgressUpdate(BaseModel):
    stage: AgentStage
    status: CompletionStatus
    files_written_or_commands_executed: list[str] = Field(default_factory=list)


class FileTaskProgress(BaseModel):
    task_id: str
    relative_path: str
    status: Literal["pending", "written_untested", "tested", "failed"] = "pending"


class WriterInput(BaseModel):
    user_prompt: str
    progress_update: AgentProgressUpdate
    task_store: list[FileTaskProgress] = Field(default_factory=list)
    supervisor_output: SupervisorOutput
    file_task: FileTask
    previous_executor_output: str | None = None


class WriterOutput(BaseModel):
    task_id: str
    relative_path: str
    status: Literal["done_untested", "completed", "failed", "unknown"] = "unknown"
    summary: str = ""
    raw_agent_state: dict[str, Any] = Field(default_factory=dict)


class ExecutorInput(BaseModel):
    user_prompt: str
    progress_update: AgentProgressUpdate
    task_store: list[FileTaskProgress] = Field(default_factory=list)
    supervisor_output: SupervisorOutput
    command_phase: Literal["setup", "test"]
    command: list[str]
    container_id: str | None = None


class ExecutorCommandResult(BaseModel):
    command_phase: Literal["setup", "test"]
    command: list[str]
    status: Literal["success", "error"]
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    message: str = ""


class ExecutorOutput(BaseModel):
    status: Literal["success", "error"]
    container_id: str | None = None
    command_results: list[ExecutorCommandResult] = Field(default_factory=list)
    summary: str = ""
    raw_agent_state: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    approved: bool
    message: str = ""


class TraceEvent(BaseModel):
    agent: str
    step: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    approval: ApprovalDecision | None = None


class MediumGraphResult(BaseModel):
    status: Literal["success", "failed_needs_human"]
    attempts: int
    current_agent: str | None = None
    progress_update: AgentProgressUpdate | None = None
    task_store: list[FileTaskProgress] = Field(default_factory=list)
    supervisor_output: SupervisorOutput
    writer_outputs: list[WriterOutput] = Field(default_factory=list)
    executor_output: ExecutorOutput | None = None
    trace: list[TraceEvent] = Field(default_factory=list)


def _normalize_commands(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        return [shlex.split(value)]
    if isinstance(value, dict):
        command = value.get("command") or value.get("cmd") or value.get("args")
        return [_normalize_command(command)] if command is not None else value
    if isinstance(value, list):
        normalized = []
        for command in value:
            normalized.append(_normalize_command(command))
        return normalized
    return value


def _normalize_command(command: Any) -> Any:
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, dict):
        command_value = command.get("command") or command.get("cmd") or command.get("args")
        return _normalize_command(command_value) if command_value is not None else command
    return command

from __future__ import annotations

from typing import Optional

from pydantic import Field

from src.evaluation.sdlc_eval_schema import (
    ComponentSpec,
    GraphDependencySpec,
    HLArchitectureSpec,
    ImplementationStep,
    ModuleSpec,
    RepositoryStructure,
    RequirementsSpec,
    StrictBaseModel,
    TestPlan,
)


class GeneratedFile(StrictBaseModel):
    path: str = ""
    content: str = ""
    purpose: Optional[str] = None


class ExecutionResult(StrictBaseModel):
    command: str = ""
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    summary: str = ""


class ToolCallRecord(StrictBaseModel):
    tool_name: str = ""
    args_summary: str = ""
    success: bool = False
    result_summary: str = ""
    error: Optional[str] = None


class SystemRunMetrics(StrictBaseModel):
    total_steps: int = 0
    agent_messages: int = 0
    tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    execution_commands: int = 0
    failed_execution_commands: int = 0
    repair_attempts: int = 0
    runtime_seconds: Optional[float] = None


class SystemRunOutput(StrictBaseModel):
    project_id: Optional[str] = None
    system_name: str = ""
    status: str = "unknown"

    requirements: RequirementsSpec = Field(default_factory=RequirementsSpec)
    high_level_design: HLArchitectureSpec = Field(default_factory=HLArchitectureSpec)

    components: list[ComponentSpec] = Field(default_factory=list)
    modules: list[ModuleSpec] = Field(default_factory=list)
    module_dependency_graph: GraphDependencySpec = Field(default_factory=GraphDependencySpec)

    repository_structure: RepositoryStructure = Field(default_factory=RepositoryStructure)
    implementation_plan: list[ImplementationStep] = Field(default_factory=list)
    test_plan: TestPlan = Field(default_factory=TestPlan)

    generated_files: list[GeneratedFile] = Field(default_factory=list)
    execution_results: list[ExecutionResult] = Field(default_factory=list)

    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    metrics: SystemRunMetrics = Field(default_factory=SystemRunMetrics)

    summary: str = ""
    limitations: list[str] = Field(default_factory=list)
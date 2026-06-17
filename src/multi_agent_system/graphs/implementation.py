from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.agents.AgentFactory import AgentFactory
from src.multi_agent_system.output_schema import (
    ComponentDecompositionOutput,
    ComponentExtractionOutput,
    ComponentImplementationPlan,
    EnvironmentSetupPlan,
    FileWriteResult,
    ImplementationExecutionResult,
    ImplementationFilePlan,
    ImplementationPlan,
    ImplementationStageOutput,
    ModuleDesignOutput,
    PlanningTask,
    PlanningStageOutput,
    RepositoryStructure,
    RequirementsStageOutput,
    TestExecutionResult,
    TestPlanNode,
)
from src.multi_agent_system.store import ProjectStoreRepository
from src.settings import settings
from src.tools.docker_code_execution.repo_tool import run_repository_command
from src.tools.file_system_tools.inspect_repository import inspect_repository


ImplementationStatus = Literal["complete", "failed", "running"]
TaskProgressStatus = Literal["pending", "completed", "failed"]
PlannerRunner = Callable[[Any, BaseModel], Any]
WriterRunner = Callable[[Any, BaseModel], Any]
ExecutorRunner = Callable[[Any, BaseModel], Any]
PlanningFeedbackCallback = Callable[[ComponentImplementationPlan, "PlanningReviewOutput", str], str]
ModelT = TypeVar("ModelT", bound=BaseModel)


WRITER_TASK_KINDS = {
    "create_file",
    "create_test_file",
    "create_integration_test",
    "review",
}
EXECUTOR_TASK_KINDS = {
    "execute_test",
    "execute_integration_test",
}


PLANNING_AGENT_PROMPT = """
You are the Planning Agent for the implementation phase of a multi-agent SDLC system.

Create a ComponentImplementationPlan for exactly one target component. The plan is a tree:
- component root
- module nodes
- explicit ordered tasks for implementation, test creation, and test execution checkpoints
- final integration test task and integration execution task

Rules:
- Return tasks, not command lists.
- For each module, create exactly this chain:
  create_file -> create_test_file -> execute_test.
- After all module chains, add:
  create_integration_test -> execute_integration_test.
- Every execute task must set test_target_task_id to the test-file task it runs.
- Use depends_on to make the ordering unambiguous.
- Descriptions must tell the Writer what to build or test, not how to run shell commands.
- Keep the plan specific to the provided component decomposition and module design.
- If previous_planning_feedback is present, treat it as mandatory revision guidance.
- The plan must include every module listed in module_design.modules unless the feedback explicitly says otherwise.
""".strip()


class PlanningAgentInput(BaseModel):
    project_prompt: str
    requirements: RequirementsStageOutput
    component_extraction: ComponentExtractionOutput
    target_component: str
    component_decomposition: ComponentDecompositionOutput
    module_design: ModuleDesignOutput
    repository_structure: RepositoryStructure | None = None
    environment_setup: EnvironmentSetupPlan | None = None
    previous_planning_feedback: str = ""


class PlanningTaskProgress(BaseModel):
    task_id: str
    kind: str
    relative_path: str | None = None
    status: TaskProgressStatus = "pending"


class WriterTaskInput(BaseModel):
    project_prompt: str
    requirements: RequirementsStageOutput
    component_extraction: ComponentExtractionOutput
    target_component: str
    component_decomposition: ComponentDecompositionOutput
    module_design: ModuleDesignOutput
    implementation_plan: ComponentImplementationPlan
    current_task: PlanningTask
    previous_executor_output: "ExecutorTaskOutput | None" = None


class WriterTaskOutput(BaseModel):
    task_id: str
    status: Literal["success", "error"] = "success"
    summary: str = ""
    raw_output: Any | None = None


class ExecutorTaskInput(BaseModel):
    project_prompt: str
    requirements: RequirementsStageOutput
    component_extraction: ComponentExtractionOutput
    target_component: str
    component_decomposition: ComponentDecompositionOutput
    module_design: ModuleDesignOutput
    implementation_plan: ComponentImplementationPlan
    current_task: PlanningTask
    test_file_task: PlanningTask | None = None
    environment_setup: EnvironmentSetupPlan | None = None


class ExecutorTaskOutput(BaseModel):
    task_id: str
    status: Literal["success", "error"] = "success"
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    message: str = ""
    summary: str = ""
    raw_output: Any | None = None


class PlanningReviewOutput(BaseModel):
    approved: bool = True
    feedback: str = ""
    required_changes: list[str] = Field(default_factory=list)


class ImplementationState(TypedDict, total=False):
    project_prompt: str
    requirements: RequirementsStageOutput
    component_extraction: ComponentExtractionOutput
    target_component: str
    component_decomposition: ComponentDecompositionOutput
    module_design: ModuleDesignOutput
    repository_structure: RepositoryStructure | None
    environment_setup: EnvironmentSetupPlan | None
    previous_planning_feedback: str
    implementation_plan: ComponentImplementationPlan
    planning_review: PlanningReviewOutput
    task_progress: list[PlanningTaskProgress]
    current_task_id: str | None
    previous_executor_output: ExecutorTaskOutput | None
    writer_outputs: list[WriterTaskOutput]
    executor_outputs: list[ExecutorTaskOutput]
    planning_revision_count: int
    max_planning_iterations: int
    repair_attempts: int
    max_repair_attempts: int
    status: ImplementationStatus
    failure_reason: str


class ImplementationRunResult(BaseModel):
    status: ImplementationStatus
    implementation_plan: ComponentImplementationPlan | None = None
    planning_review: PlanningReviewOutput | None = None
    task_progress: list[PlanningTaskProgress] = Field(default_factory=list)
    writer_outputs: list[WriterTaskOutput] = Field(default_factory=list)
    executor_outputs: list[ExecutorTaskOutput] = Field(default_factory=list)
    planning_revision_count: int = 0
    failure_reason: str = ""
    repair_attempts: int = 0


ExecutionStatus = Literal["complete", "failed", "running"]
ImplementationFileStatus = Literal["pending", "written", "failed"]
ExecutionWriterRunner = Callable[[Any, BaseModel], Any]
RepositoryFileWriter = Callable[[str, str], FileWriteResult]
RepositoryCommandExecutor = Callable[[list[str], int], Any]


IMPLEMENTATION_EXECUTION_WRITER_PROMPT = """
You are the Implementation Writer in a deterministic SDLC execution graph.

Return exactly one ImplementationWriterOutput for the requested file.
The graph will write the file and run Docker commands deterministically.

Rules:
- Return full file content, not a patch.
- Use the planning file target, related test nodes, requirements, and design context.
- If previous_test_failure is present, repair only the requested file while preserving intended behavior.
- Do not claim that you executed commands or wrote files.
""".strip()


class ImplementationWriterInput(BaseModel):
    project_prompt: str
    requirements: RequirementsStageOutput | None = None
    component_extraction: ComponentExtractionOutput | None = None
    component_decompositions: dict[str, ComponentDecompositionOutput] = Field(default_factory=dict)
    module_designs: dict[str, ModuleDesignOutput] = Field(default_factory=dict)
    planning_output: PlanningStageOutput
    current_file: ImplementationFilePlan
    related_test_nodes: list[TestPlanNode] = Field(default_factory=list)
    repository_tree: str = ""
    previous_test_failure: TestExecutionResult | None = None


class ImplementationWriterOutput(BaseModel):
    relative_path: str
    content: str
    target_modules: list[str] = Field(default_factory=list)
    target_test_nodes: list[str] = Field(default_factory=list)
    summary: str = ""


class ImplementationFileProgress(BaseModel):
    file_id: str
    relative_path: str
    status: ImplementationFileStatus = "pending"
    repair_attempts: int = 0


class ImplementationExecutionState(TypedDict, total=False):
    project_id: str
    project_prompt: str
    requirements: RequirementsStageOutput | None
    component_extraction: ComponentExtractionOutput | None
    component_decompositions: dict[str, ComponentDecompositionOutput]
    module_designs: dict[str, ModuleDesignOutput]
    repository_structure: RepositoryStructure | None
    environment_setup: EnvironmentSetupPlan | None
    planning_output: PlanningStageOutput
    repository_tree: str
    file_progress: list[ImplementationFileProgress]
    current_file_id: str | None
    current_writer_output: ImplementationWriterOutput | None
    previous_test_failure: TestExecutionResult | None
    file_writes: list[FileWriteResult]
    test_executions: list[TestExecutionResult]
    max_repair_attempts: int
    command_timeout_s: int
    stage_output: ImplementationStageOutput
    status: ExecutionStatus
    failure_reason: str


class ImplementationExecutionRunResult(BaseModel):
    status: ExecutionStatus
    stage_output: ImplementationStageOutput | None = None
    file_writes: list[FileWriteResult] = Field(default_factory=list)
    test_executions: list[TestExecutionResult] = Field(default_factory=list)
    failure_reason: str = ""


class ImplementationExecutionGraph:
    """Executes approved planning.json with deterministic file writes and Docker checks."""

    def __init__(
        self,
        writer_agent: Any | None = None,
        writer_runner: ExecutionWriterRunner | None = None,
        repository_inspector: Callable[[], Any] | None = None,
        file_writer: RepositoryFileWriter | None = None,
        command_executor: RepositoryCommandExecutor | None = None,
        store_repository: ProjectStoreRepository | None = None,
        model_name: str = "gpt-5.4-mini"
,
        temperature: float = 0.2,
        command_timeout_s: int = 120,
        max_repair_attempts: int = 3,
        verbose: bool = True,
    ) -> None:
        self.writer_agent = writer_agent
        self.writer_runner = writer_runner or _run_structured_agent
        self.repository_inspector = repository_inspector or _inspect_repository_root
        self.file_writer = file_writer or _write_repository_file
        self.command_executor = command_executor or _execute_repository_command
        self.store_repository = store_repository
        self.model_name = model_name
        self.temperature = temperature
        self.command_timeout_s = command_timeout_s
        self.max_repair_attempts = max_repair_attempts
        self.verbose = verbose
        self.graph = self._build_execution_graph()

    def run(
        self,
        project_id: str,
        project_prompt: str,
        requirements: RequirementsStageOutput | None,
        component_extraction: ComponentExtractionOutput | None,
        component_decompositions: dict[str, ComponentDecompositionOutput],
        module_designs: dict[str, ModuleDesignOutput],
        planning_output: PlanningStageOutput,
        repository_structure: RepositoryStructure | None = None,
        environment_setup: EnvironmentSetupPlan | None = None,
    ) -> ImplementationExecutionRunResult:
        initial_state: ImplementationExecutionState = {
            "project_id": project_id,
            "project_prompt": project_prompt,
            "requirements": requirements,
            "component_extraction": component_extraction,
            "component_decompositions": component_decompositions,
            "module_designs": module_designs,
            "repository_structure": repository_structure,
            "environment_setup": environment_setup,
            "planning_output": planning_output,
            "file_progress": [
                ImplementationFileProgress(
                    file_id=file_plan.file_id,
                    relative_path=file_plan.relative_path,
                )
                for file_plan in planning_output.implementation_plan.files
            ],
            "current_file_id": None,
            "previous_test_failure": None,
            "file_writes": [],
            "test_executions": [],
            "max_repair_attempts": self.max_repair_attempts,
            "command_timeout_s": self.command_timeout_s,
            "status": "running",
        }
        try:
            final_state = self.graph.invoke(initial_state)
            return self._to_result(final_state)
        except Exception as error:
            return ImplementationExecutionRunResult(
                status="failed",
                failure_reason=_format_exception(error),
            )

    def _build_execution_graph(self):
        graph_builder = StateGraph(ImplementationExecutionState)
        graph_builder.add_node("inspect_repository", self.inspect_repository)
        graph_builder.add_node("select_next_action", self.select_next_action)
        graph_builder.add_node("code_writer", self.code_writer)
        graph_builder.add_node("deterministic_file_write", self.deterministic_file_write)
        graph_builder.add_node("docker_test_executor", self.docker_test_executor)
        graph_builder.add_node("repair_or_continue", self.repair_or_continue)
        graph_builder.add_node("final_verification", self.final_verification)
        graph_builder.add_node("complete", self.complete)
        graph_builder.add_node("failed", self.failed)

        graph_builder.add_edge(START, "inspect_repository")
        graph_builder.add_edge("inspect_repository", "select_next_action")
        graph_builder.add_conditional_edges(
            "select_next_action",
            self.route_after_select_next_action,
            {
                "code_writer": "code_writer",
                "final_verification": "final_verification",
                "failed": "failed",
            },
        )
        graph_builder.add_edge("code_writer", "deterministic_file_write")
        graph_builder.add_conditional_edges(
            "deterministic_file_write",
            self.route_after_file_write,
            {
                "docker_test_executor": "docker_test_executor",
                "failed": "failed",
            },
        )
        graph_builder.add_edge("docker_test_executor", "repair_or_continue")
        graph_builder.add_conditional_edges(
            "repair_or_continue",
            self.route_after_repair_or_continue,
            {
                "code_writer": "code_writer",
                "select_next_action": "select_next_action",
                "failed": "failed",
            },
        )
        graph_builder.add_conditional_edges(
            "final_verification",
            self.route_after_final_verification,
            {
                "complete": "complete",
                "failed": "failed",
            },
        )
        graph_builder.add_edge("complete", END)
        graph_builder.add_edge("failed", END)
        return graph_builder.compile()

    def inspect_repository(self, state: ImplementationExecutionState) -> ImplementationExecutionState:
        self._announce_node("inspect_repository")
        return {
            **state,
            "repository_tree": _extract_repository_tree(self.repository_inspector()),
        }

    def select_next_action(self, state: ImplementationExecutionState) -> ImplementationExecutionState:
        self._announce_node("select_next_action")
        next_progress = _next_pending_file(state.get("file_progress", []))
        return {
            **state,
            "current_file_id": next_progress.file_id if next_progress else None,
        }

    @staticmethod
    def route_after_select_next_action(state: ImplementationExecutionState) -> str:
        if state.get("status") == "failed":
            return "failed"
        if state.get("current_file_id") is None:
            return "final_verification"
        return "code_writer"

    def code_writer(self, state: ImplementationExecutionState) -> ImplementationExecutionState:
        self._announce_node("code_writer")
        current_file = _require_current_file(state)
        writer_input = ImplementationWriterInput(
            project_prompt=state["project_prompt"],
            requirements=state.get("requirements"),
            component_extraction=state.get("component_extraction"),
            component_decompositions=state.get("component_decompositions", {}),
            module_designs=state.get("module_designs", {}),
            planning_output=state["planning_output"],
            current_file=current_file,
            related_test_nodes=_related_test_nodes(
                current_file,
                state["planning_output"].test_plan.nodes,
            ),
            repository_tree=state.get("repository_tree", ""),
            previous_test_failure=state.get("previous_test_failure"),
        )
        raw_output = self.writer_runner(self._get_writer_agent(), writer_input)
        writer_output = _coerce_model(raw_output, ImplementationWriterOutput)
        if writer_output.relative_path != current_file.relative_path:
            writer_output = writer_output.model_copy(update={"relative_path": current_file.relative_path})
        return {
            **state,
            "current_writer_output": writer_output,
        }

    def deterministic_file_write(self, state: ImplementationExecutionState) -> ImplementationExecutionState:
        self._announce_node("deterministic_file_write")
        writer_output = state["current_writer_output"]
        if writer_output is None:
            return {
                **state,
                "status": "failed",
                "failure_reason": "Writer output was missing.",
            }
        write_result = self.file_writer(writer_output.relative_path, writer_output.content)
        file_writes = [*state.get("file_writes", []), write_result]
        if write_result.status != "success":
            return {
                **state,
                "file_writes": file_writes,
                "status": "failed",
                "failure_reason": write_result.message or f"Failed to write {writer_output.relative_path}",
            }
        return {
            **state,
            "file_writes": file_writes,
            "file_progress": _update_file_status(
                state.get("file_progress", []),
                state.get("current_file_id"),
                "written",
            ),
        }

    @staticmethod
    def route_after_file_write(state: ImplementationExecutionState) -> str:
        if state.get("status") == "failed":
            return "failed"
        return "docker_test_executor"

    def docker_test_executor(self, state: ImplementationExecutionState) -> ImplementationExecutionState:
        self._announce_node("docker_test_executor")
        current_file = _require_current_file(state)
        command = _targeted_test_command(current_file, state["planning_output"])
        if command is None:
            return {
                **state,
                "previous_test_failure": None,
            }
        result = _coerce_test_execution_result(
            self.command_executor(command, state.get("command_timeout_s", self.command_timeout_s)),
            command,
        )
        test_executions = [*state.get("test_executions", []), result]
        if result.status != "success":
            return {
                **state,
                "test_executions": test_executions,
                "previous_test_failure": result,
            }
        return {
            **state,
            "test_executions": test_executions,
            "previous_test_failure": None,
        }

    def repair_or_continue(self, state: ImplementationExecutionState) -> ImplementationExecutionState:
        self._announce_node("repair_or_continue")
        if state.get("previous_test_failure") is None:
            return state
        current_file_id = state.get("current_file_id")
        progress = _find_file_progress(state.get("file_progress", []), current_file_id)
        repair_attempts = (progress.repair_attempts if progress else 0) + 1
        if repair_attempts > state.get("max_repair_attempts", self.max_repair_attempts):
            return {
                **state,
                "file_progress": _update_file_status(
                    state.get("file_progress", []),
                    current_file_id,
                    "failed",
                    repair_attempts=repair_attempts,
                ),
                "status": "failed",
                "failure_reason": (
                    f"Maximum repair attempts exceeded for {current_file_id}: "
                    f"{repair_attempts}/{state.get('max_repair_attempts', self.max_repair_attempts)}"
                ),
            }
        return {
            **state,
            "file_progress": _update_file_status(
                state.get("file_progress", []),
                current_file_id,
                "pending",
                repair_attempts=repair_attempts,
            ),
        }

    @staticmethod
    def route_after_repair_or_continue(state: ImplementationExecutionState) -> str:
        if state.get("status") == "failed":
            return "failed"
        if state.get("previous_test_failure") is not None:
            return "code_writer"
        return "select_next_action"

    def final_verification(self, state: ImplementationExecutionState) -> ImplementationExecutionState:
        self._announce_node("final_verification")
        commands = _final_verification_commands(
            state["planning_output"],
            state.get("environment_setup"),
        )
        test_executions = list(state.get("test_executions", []))
        for command in commands:
            result = _coerce_test_execution_result(
                self.command_executor(command, state.get("command_timeout_s", self.command_timeout_s)),
                command,
            )
            test_executions.append(result)
            if result.status != "success":
                return {
                    **state,
                    "test_executions": test_executions,
                    "status": "failed",
                    "failure_reason": result.summary or result.stderr or f"Final verification failed: {' '.join(command)}",
                }
        return {
            **state,
            "test_executions": test_executions,
            "status": "complete",
        }

    @staticmethod
    def route_after_final_verification(state: ImplementationExecutionState) -> str:
        if state.get("status") == "complete":
            return "complete"
        return "failed"

    def complete(self, state: ImplementationExecutionState) -> ImplementationExecutionState:
        self._announce_node("complete")
        stage_output = _build_implementation_stage_output(state, approved=True)
        self._save_implementation(state, stage_output)
        return {
            **state,
            "stage_output": stage_output,
            "status": "complete",
        }

    def failed(self, state: ImplementationExecutionState) -> ImplementationExecutionState:
        self._announce_node("failed")
        stage_output = _build_implementation_stage_output(state, approved=False)
        self._save_implementation(state, stage_output)
        return {
            **state,
            "stage_output": stage_output,
            "status": "failed",
        }

    def _get_writer_agent(self) -> Any:
        if self.writer_agent is None:
            self.writer_agent = AgentFactory.build_agent(
                prompt=IMPLEMENTATION_EXECUTION_WRITER_PROMPT,
                tools=[],
                temperature=self.temperature,
                model_name=self.model_name,
                response_format=ImplementationWriterOutput,
            )
        return self.writer_agent

    def _save_implementation(
        self,
        state: ImplementationExecutionState,
        stage_output: ImplementationStageOutput,
    ) -> None:
        if self.store_repository is None or not state.get("project_id"):
            return
        self.store_repository.save_implementation(state["project_id"], stage_output)

    def _announce_node(self, node_name: str) -> None:
        if self.verbose:
            print(f"Implementation execution node: {node_name}")

    @staticmethod
    def _to_result(state: ImplementationExecutionState) -> ImplementationExecutionRunResult:
        return ImplementationExecutionRunResult(
            status=state.get("status", "failed"),
            stage_output=state.get("stage_output"),
            file_writes=state.get("file_writes", []),
            test_executions=state.get("test_executions", []),
            failure_reason=state.get("failure_reason", ""),
        )


class ImplementationPlanningGraph:
    """Planning Agent -> Writer -> Executor graph for one component."""

    def __init__(
        self,
        planning_agent: Any | None = None,
        writer_agent: Any | None = None,
        executor_agent: Any | None = None,
        planner_runner: PlannerRunner | None = None,
        writer_runner: WriterRunner | None = None,
        executor_runner: ExecutorRunner | None = None,
        planning_feedback_callback: PlanningFeedbackCallback | None = None,
        model_name: str = "gpt-5.4-mini"
,
        temperature: float = 0.2,
    ) -> None:
        self.planning_agent = planning_agent
        self.writer_agent = writer_agent
        self.executor_agent = executor_agent
        self.planner_runner = planner_runner or _run_structured_agent
        self.writer_runner = writer_runner or _run_structured_agent
        self.executor_runner = executor_runner or _run_structured_agent
        self.planning_feedback_callback = planning_feedback_callback
        self.model_name = model_name
        self.temperature = temperature
        self.graph = self._build_graph()

    def run(
        self,
        project_prompt: str,
        requirements: RequirementsStageOutput,
        component_extraction: ComponentExtractionOutput,
        target_component: str,
        component_decomposition: ComponentDecompositionOutput,
        module_design: ModuleDesignOutput,
        repository_structure: RepositoryStructure | None = None,
        environment_setup: EnvironmentSetupPlan | None = None,
        previous_planning_feedback: str = "",
        max_planning_iterations: int = 3,
        max_repair_attempts: int = 5,
    ) -> ImplementationRunResult:
        initial_state: ImplementationState = {
            "project_prompt": project_prompt,
            "requirements": requirements,
            "component_extraction": component_extraction,
            "target_component": target_component,
            "component_decomposition": component_decomposition,
            "module_design": module_design,
            "repository_structure": repository_structure,
            "environment_setup": environment_setup,
            "previous_planning_feedback": previous_planning_feedback,
            "task_progress": [],
            "current_task_id": None,
            "previous_executor_output": None,
            "writer_outputs": [],
            "executor_outputs": [],
            "planning_revision_count": 0,
            "max_planning_iterations": max_planning_iterations,
            "repair_attempts": 0,
            "max_repair_attempts": max_repair_attempts,
            "status": "running",
        }
        try:
            final_state = self.graph.invoke(initial_state)
            return self._to_result(final_state)
        except Exception as error:
            return ImplementationRunResult(
                status="failed",
                failure_reason=_format_exception(error),
                repair_attempts=initial_state.get("repair_attempts", 0),
            )

    def _build_graph(self):
        graph_builder = StateGraph(ImplementationState)
        graph_builder.add_node("planning_agent", self.planning_agent_node)
        graph_builder.add_node("planning_review", self.planning_review_node)
        graph_builder.add_node("writer", self.writer_node)
        graph_builder.add_node("executor", self.executor_node)
        graph_builder.add_node("complete", self.complete_node)
        graph_builder.add_node("failed", self.failed_node)

        graph_builder.add_edge(START, "planning_agent")
        graph_builder.add_edge("planning_agent", "planning_review")
        graph_builder.add_conditional_edges(
            "planning_review",
            self.route_after_planning_review,
            {
                "writer": "writer",
                "planning_agent": "planning_agent",
                "failed": "failed",
            },
        )
        graph_builder.add_conditional_edges(
            "writer",
            self.route_after_writer,
            {
                "writer": "writer",
                "executor": "executor",
                "complete": "complete",
                "failed": "failed",
            },
        )
        graph_builder.add_conditional_edges(
            "executor",
            self.route_after_executor,
            {
                "writer": "writer",
                "executor": "executor",
                "complete": "complete",
                "failed": "failed",
            },
        )
        graph_builder.add_edge("complete", END)
        graph_builder.add_edge("failed", END)
        return graph_builder.compile()

    def planning_agent_node(self, state: ImplementationState) -> ImplementationState:
        planner_input = PlanningAgentInput(
            project_prompt=state["project_prompt"],
            requirements=state["requirements"],
            component_extraction=state["component_extraction"],
            target_component=state["target_component"],
            component_decomposition=state["component_decomposition"],
            module_design=state["module_design"],
            repository_structure=state.get("repository_structure"),
            environment_setup=state.get("environment_setup"),
            previous_planning_feedback=state.get("previous_planning_feedback", ""),
        )
        raw_output = self.planner_runner(self._get_planning_agent(), planner_input)
        implementation_plan = _coerce_model(raw_output, ComponentImplementationPlan)
        return {
            **state,
            "implementation_plan": implementation_plan,
            "current_task_id": None,
            "task_progress": [],
        }

    def planning_review_node(self, state: ImplementationState) -> ImplementationState:
        review = _review_component_plan(
            state["implementation_plan"],
            state["target_component"],
            state["module_design"],
        )
        if not review.approved:
            revision_count = state.get("planning_revision_count", 0) + 1
            review_feedback = _format_review_failure(review)
            human_feedback = self._get_human_planning_feedback(
                state["implementation_plan"],
                review,
                review_feedback,
            )
            next_feedback = _append_planning_feedback(
                state.get("previous_planning_feedback", ""),
                _format_planning_revision_feedback(
                    state["implementation_plan"],
                    review,
                    human_feedback,
                ),
            )
            if revision_count >= state.get("max_planning_iterations", 3):
                return {
                    **state,
                    "planning_review": review,
                    "previous_planning_feedback": next_feedback,
                    "planning_revision_count": revision_count,
                    "status": "failed",
                    "failure_reason": (
                        f"{review_feedback} Planning iteration limit reached "
                        f"({revision_count}/{state.get('max_planning_iterations', 3)})."
                    ),
                }
            return {
                **state,
                "planning_review": review,
                "previous_planning_feedback": next_feedback,
                "planning_revision_count": revision_count,
                "current_task_id": None,
                "task_progress": [],
                "status": "running",
            }
        plan = state["implementation_plan"].model_copy(update={"approved": True})
        next_state: ImplementationState = {
            **state,
            "implementation_plan": plan,
            "planning_review": review,
            "task_progress": _build_task_progress(plan),
        }
        next_task = _next_ready_task(next_state)
        next_state["current_task_id"] = next_task.task_id if next_task else None
        return next_state

    def writer_node(self, state: ImplementationState) -> ImplementationState:
        task = _require_current_task(state)
        writer_input = WriterTaskInput(
            project_prompt=state["project_prompt"],
            requirements=state["requirements"],
            component_extraction=state["component_extraction"],
            target_component=state["target_component"],
            component_decomposition=state["component_decomposition"],
            module_design=state["module_design"],
            implementation_plan=state["implementation_plan"],
            current_task=task,
            previous_executor_output=state.get("previous_executor_output"),
        )
        raw_output = self.writer_runner(self._get_writer_agent(), writer_input)
        writer_output = _coerce_writer_output(raw_output, task)
        next_state: ImplementationState = {
            **state,
            "writer_outputs": [*state.get("writer_outputs", []), writer_output],
        }
        if writer_output.status == "error":
            return {
                **next_state,
                "task_progress": _update_task_status(state.get("task_progress", []), task.task_id, "failed"),
                "status": "failed",
                "failure_reason": writer_output.summary or f"Writer failed task {task.task_id}",
            }
        next_state = {
            **next_state,
            "task_progress": _update_task_status(state.get("task_progress", []), task.task_id, "completed"),
            "previous_executor_output": None,
        }
        next_task = _next_ready_task(next_state)
        next_state["current_task_id"] = next_task.task_id if next_task else None
        return next_state

    def executor_node(self, state: ImplementationState) -> ImplementationState:
        task = _require_current_task(state)
        test_file_task = _find_task(state["implementation_plan"], task.test_target_task_id)
        executor_input = ExecutorTaskInput(
            project_prompt=state["project_prompt"],
            requirements=state["requirements"],
            component_extraction=state["component_extraction"],
            target_component=state["target_component"],
            component_decomposition=state["component_decomposition"],
            module_design=state["module_design"],
            implementation_plan=state["implementation_plan"],
            current_task=task,
            test_file_task=test_file_task,
            environment_setup=state.get("environment_setup"),
        )
        raw_output = self.executor_runner(self._get_executor_agent(), executor_input)
        executor_output = _coerce_executor_output(raw_output, task)
        executor_outputs = [*state.get("executor_outputs", []), executor_output]
        if executor_output.status == "success":
            next_state = {
                **state,
                "executor_outputs": executor_outputs,
                "task_progress": _update_task_status(state.get("task_progress", []), task.task_id, "completed"),
                "previous_executor_output": None,
            }
            next_task = _next_ready_task(next_state)
            next_state["current_task_id"] = next_task.task_id if next_task else None
            return next_state

        repair_attempts = state.get("repair_attempts", 0) + 1
        if repair_attempts > state.get("max_repair_attempts", 5):
            return {
                **state,
                "executor_outputs": executor_outputs,
                "repair_attempts": repair_attempts,
                "previous_executor_output": executor_output,
                "status": "failed",
                "failure_reason": executor_output.summary or f"Executor failed task {task.task_id}",
            }

        repair_task_id = _repair_task_id_for_execution_failure(state["implementation_plan"], task)
        task_progress = state.get("task_progress", [])
        if repair_task_id:
            task_progress = _update_task_status(task_progress, repair_task_id, "failed")
        return {
            **state,
            "executor_outputs": executor_outputs,
            "task_progress": task_progress,
            "current_task_id": repair_task_id,
            "previous_executor_output": executor_output,
            "repair_attempts": repair_attempts,
        }

    @staticmethod
    def route_after_planning_review(state: ImplementationState) -> str:
        if state.get("status") == "failed":
            return "failed"
        review = state.get("planning_review")
        if review is not None and not review.approved:
            return "planning_agent"
        return _route_by_current_task(state)

    @staticmethod
    def route_after_writer(state: ImplementationState) -> str:
        return _route_by_current_task(state)

    @staticmethod
    def route_after_executor(state: ImplementationState) -> str:
        return _route_by_current_task(state)

    @staticmethod
    def complete_node(state: ImplementationState) -> ImplementationState:
        return {
            **state,
            "status": "complete",
        }

    @staticmethod
    def failed_node(state: ImplementationState) -> ImplementationState:
        return {
            **state,
            "status": "failed",
            "failure_reason": state.get("failure_reason", "Implementation planning graph failed."),
        }

    def _get_planning_agent(self) -> Any:
        if self.planning_agent is None:
            self.planning_agent = AgentFactory.build_agent(
                prompt=PLANNING_AGENT_PROMPT,
                tools=[],
                temperature=self.temperature,
                model_name=self.model_name,
                response_format=ComponentImplementationPlan,
            )
        return self.planning_agent

    def _get_writer_agent(self) -> Any:
        if self.writer_agent is None:
            self.writer_agent = AgentFactory.build_agent(
                prompt="You are the Writer Agent. Complete only the current PlanningTask.",
                tools=[],
                temperature=self.temperature,
                model_name=self.model_name,
                response_format=WriterTaskOutput,
            )
        return self.writer_agent

    def _get_executor_agent(self) -> Any:
        if self.executor_agent is None:
            self.executor_agent = AgentFactory.build_agent(
                prompt="You are the Executor Agent. Execute only the current explicit test checkpoint.",
                tools=[],
                temperature=self.temperature,
                model_name=self.model_name,
                response_format=ExecutorTaskOutput,
            )
        return self.executor_agent

    @staticmethod
    def _to_result(state: ImplementationState) -> ImplementationRunResult:
        return ImplementationRunResult(
            status=state.get("status", "failed"),
            implementation_plan=state.get("implementation_plan"),
            planning_review=state.get("planning_review"),
            task_progress=state.get("task_progress", []),
            writer_outputs=state.get("writer_outputs", []),
            executor_outputs=state.get("executor_outputs", []),
            failure_reason=state.get("failure_reason", ""),
            planning_revision_count=state.get("planning_revision_count", 0),
            repair_attempts=state.get("repair_attempts", 0),
        )

    def _get_human_planning_feedback(
        self,
        plan: ComponentImplementationPlan,
        review: PlanningReviewOutput,
        review_feedback: str,
    ) -> str:
        if self.planning_feedback_callback is None:
            return ""
        return self.planning_feedback_callback(plan, review, review_feedback).strip()


def build_implementation_graph(**kwargs: Any) -> ImplementationPlanningGraph:
    return ImplementationPlanningGraph(**kwargs)


def build_implementation_execution_graph(**kwargs: Any) -> ImplementationExecutionGraph:
    return ImplementationExecutionGraph(**kwargs)


def _load_execution_inputs_from_store(
    project_id: str,
    store_repository: ProjectStoreRepository | None = None,
) -> dict[str, Any]:
    repository = store_repository or ProjectStoreRepository()
    store = repository.load_project(project_id)
    planning_output = repository.load_planning(project_id)

    if store.requirements is None:
        raise RuntimeError(f"Project {project_id} does not have approved requirements in the store.")
    if store.component_extraction is None:
        raise RuntimeError(f"Project {project_id} does not have component extraction in the store.")
    if not store.module_designs:
        raise RuntimeError(f"Project {project_id} does not have module designs in the store.")

    return {
        "project_id": store.project_id,
        "project_prompt": store.project_prompt,
        "requirements": store.requirements,
        "component_extraction": store.component_extraction,
        "component_decompositions": store.component_decompositions,
        "module_designs": store.module_designs,
        "planning_output": planning_output,
        "repository_structure": store.repository_structure,
        "environment_setup": store.environment_setup,
    }


def _print_loaded_execution_inputs(inputs: dict[str, Any]) -> None:
    planning_output = inputs["planning_output"]
    print("\nLoaded implementation execution inputs")
    print(f"Project id: {inputs['project_id']}")
    print(f"Project prompt: {inputs['project_prompt']}")
    print(f"Implementation files: {len(planning_output.implementation_plan.files)}")
    print(f"Implementation steps: {len(planning_output.implementation_plan.steps)}")
    print(f"Test plan nodes: {len(planning_output.test_plan.nodes)}")
    print(f"Test commands: {len(planning_output.test_plan.commands)}")
    print(f"Module designs: {len(inputs.get('module_designs', {}))}")
    print(f"Environment setup loaded: {inputs.get('environment_setup') is not None}")


def _print_execution_result(result: ImplementationExecutionRunResult) -> None:
    print("\nImplementation execution result")
    print(f"Status: {result.status}")
    if result.failure_reason:
        print(f"Failure reason: {result.failure_reason}")
    print(f"File writes: {len(result.file_writes)}")
    for file_write in result.file_writes:
        print(f"- {file_write.relative_path}: {file_write.status}")
    print(f"Test executions: {len(result.test_executions)}")
    for test_execution in result.test_executions:
        print(f"- {' '.join(test_execution.command)}: {test_execution.status}")


def _load_implementation_inputs_from_store(
    project_id: str = "todo_cli",
    target_component: str | None = None,
    store_repository: ProjectStoreRepository | None = None,
) -> dict[str, Any]:
    repository = store_repository or ProjectStoreRepository()
    store = repository.load_project(project_id)

    if store.requirements is None:
        raise RuntimeError(f"Project {project_id} does not have approved requirements in the store.")
    if store.component_extraction is None:
        raise RuntimeError(f"Project {project_id} does not have component extraction in the store.")

    component_name = target_component or _default_component_name(store.component_decompositions)
    if component_name is None:
        raise RuntimeError(f"Project {project_id} does not have component decompositions in the store.")

    component_decomposition = store.component_decompositions.get(component_name)
    if component_decomposition is None:
        available = _format_available_components(store.component_decompositions)
        raise RuntimeError(
            f"Component decomposition for {component_name!r} was not found. Available components: {available}"
        )

    module_design = store.module_designs.get(component_name)
    if module_design is None:
        available = _format_available_components(store.module_designs)
        raise RuntimeError(
            f"Module design for {component_name!r} was not found. Available components: {available}"
        )

    return {
        "project_id": store.project_id,
        "project_prompt": store.project_prompt,
        "requirements": store.requirements,
        "component_extraction": store.component_extraction,
        "target_component": component_name,
        "component_decomposition": component_decomposition,
        "module_design": module_design,
        "repository_structure": store.repository_structure,
        "environment_setup": store.environment_setup,
    }


def _print_loaded_implementation_inputs(inputs: dict[str, Any]) -> None:
    requirements = inputs["requirements"].requirements
    component_extraction = inputs["component_extraction"]
    component_decomposition = inputs["component_decomposition"]
    module_design = inputs["module_design"]
    environment_setup = inputs.get("environment_setup")

    print("\nLoaded implementation planning inputs")
    print(f"Project id: {inputs.get('project_id')}")
    print(f"Project prompt: {inputs.get('project_prompt')}")
    print(f"Target component: {inputs.get('target_component')}")
    print(f"Functional requirements: {len(requirements.functional_requirements)}")
    print(f"Non-functional requirements: {len(requirements.non_functional_requirements)}")
    print(f"HLL components: {len(component_extraction.components)}")
    print(f"Component decomposition modules: {len(component_decomposition.modules)}")
    print(f"Module design modules: {len(module_design.modules)}")
    print(f"Repository structure loaded: {inputs.get('repository_structure') is not None}")
    print(f"Environment setup loaded: {environment_setup is not None}")

    if module_design.modules:
        print("\nModules for planning:")
        for module in module_design.modules:
            print(f"- {module.name}")


def _print_implementation_result(
    result: ImplementationRunResult,
    module_design: ModuleDesignOutput | None = None,
) -> None:
    print("\nImplementation planning graph result")
    print(f"Status: {result.status}")
    if result.failure_reason:
        print(f"Failure reason: {result.failure_reason}")

    if result.implementation_plan is not None:
        plan = result.implementation_plan
        print(f"Plan component: {plan.component_name}")
        print(f"Plan approved: {plan.approved}")
        print(f"Tasks: {len(plan.tasks)}")
        if plan.summary:
            print(f"Summary: {plan.summary}")
        _print_plan_debug(plan, module_design)

    if result.planning_review is not None:
        print(f"Planning review approved: {result.planning_review.approved}")
        if result.planning_review.required_changes:
            print("Required changes:")
            for change in result.planning_review.required_changes:
                print(f"- {change}")

    if result.task_progress:
        print("\nTask progress:")
        for progress in result.task_progress:
            path = f" ({progress.relative_path})" if progress.relative_path else ""
            print(f"- {progress.task_id}: {progress.status}{path}")


def _default_component_name(component_outputs: dict[str, Any]) -> str | None:
    if not component_outputs:
        return None
    return sorted(component_outputs.keys())[0]


def _format_available_components(component_outputs: dict[str, Any]) -> str:
    if not component_outputs:
        return "none"
    return ", ".join(sorted(component_outputs.keys()))


def _print_plan_debug(
    plan: ComponentImplementationPlan,
    module_design: ModuleDesignOutput | None = None,
) -> None:
    print("\nPlanning debug")
    if module_design is not None:
        expected_modules = [module.name for module in module_design.modules]
        planned_modules = [node.module_name for node in plan.module_nodes]
        missing_modules = [name for name in expected_modules if name not in planned_modules]
        extra_modules = [name for name in planned_modules if name not in expected_modules]
        print(f"Expected module nodes ({len(expected_modules)}): {', '.join(expected_modules) or 'None'}")
        print(f"Planned module nodes ({len(planned_modules)}): {', '.join(planned_modules) or 'None'}")
        print(f"Missing module nodes: {', '.join(missing_modules) or 'None'}")
        print(f"Extra module nodes: {', '.join(extra_modules) or 'None'}")

    print(f"Integration test task id: {plan.integration_test_task_id or 'None'}")
    print(f"Execute integration test task id: {plan.execute_integration_test_task_id or 'None'}")

    task_kind_counts: dict[str, int] = {}
    for task in plan.tasks:
        task_kind_counts[task.kind] = task_kind_counts.get(task.kind, 0) + 1
    print("Task kind counts:")
    for kind, count in sorted(task_kind_counts.items()):
        print(f"- {kind}: {count}")

    print("\nModule node references:")
    if not plan.module_nodes:
        print("- None")
    for node in plan.module_nodes:
        print(
            "- "
            f"{node.module_name}: "
            f"impl={node.implementation_task_id}, "
            f"test={node.test_task_id}, "
            f"execute={node.execute_test_task_id}"
        )

    print("\nTask inventory:")
    if not plan.tasks:
        print("- None")
    for task in plan.tasks:
        target = f", module={task.target_module}" if task.target_module else ""
        path = f", path={task.relative_path}" if task.relative_path else ""
        depends_on = ", ".join(task.depends_on) or "None"
        test_target = task.test_target_task_id or "None"
        print(
            "- "
            f"{task.task_id}: kind={task.kind}"
            f"{target}"
            f"{path}, depends_on={depends_on}, test_target={test_target}"
        )


def _interactive_planning_feedback(
    plan: ComponentImplementationPlan,
    review: PlanningReviewOutput,
    review_feedback: str,
) -> str:
    print("\nPlanning review failed")
    print(review_feedback)
    _print_plan_debug(plan)
    if review.required_changes:
        print("\nRequired changes:")
        for change in review.required_changes:
            print(f"- {change}")
    return input("\nAdvice for the Planning Agent before replanning (optional): ").strip()


def main() -> None:
    project_id = input("Project id to run implementation execution graph [todo_cli]: ").strip() or "todo_cli"
    store_repository = ProjectStoreRepository()
    inputs = _load_execution_inputs_from_store(
        project_id=project_id,
        store_repository=store_repository,
    )
    _print_loaded_execution_inputs(inputs)

    choice = input("\nRun implementation execution graph with these inputs? [y/N]: ").strip().lower()
    if choice not in {"y", "yes"}:
        return

    graph_inputs = {key: value for key, value in inputs.items() if key != "project_id"}
    result = build_implementation_execution_graph(
        store_repository=store_repository,
    ).run(project_id=project_id, **graph_inputs)
    _print_execution_result(result)


def _run_structured_agent(agent: Any, agent_input: BaseModel) -> Any:
    return agent.invoke(
        {
            "messages": [
                HumanMessage(content=agent_input.model_dump_json(indent=2))
            ]
        }
    )


def _coerce_model(value: Any, model_cls: type[ModelT]) -> ModelT:
    if isinstance(value, model_cls):
        return value
    if isinstance(value, dict):
        structured_response = value.get("structured_response")
        if structured_response is not None:
            return _coerce_model(structured_response, model_cls)
        return model_cls.model_validate(value)
    if isinstance(value, str):
        return model_cls.model_validate_json(value)
    if isinstance(value, BaseMessage):
        return _coerce_model(value.content, model_cls)
    raise TypeError(f"Cannot convert {type(value).__name__} to {model_cls.__name__}")


def _coerce_writer_output(value: Any, task: PlanningTask) -> WriterTaskOutput:
    if isinstance(value, WriterTaskOutput):
        output = value
    elif isinstance(value, dict):
        structured_response = value.get("structured_response")
        if structured_response is not None:
            return _coerce_writer_output(structured_response, task)
        output = WriterTaskOutput.model_validate(value)
    elif isinstance(value, str):
        output = WriterTaskOutput(task_id=task.task_id, status="success", summary=value)
    else:
        output = WriterTaskOutput(task_id=task.task_id, status="success", raw_output=value)
    if not output.task_id:
        output.task_id = task.task_id
    return output


def _coerce_executor_output(value: Any, task: PlanningTask) -> ExecutorTaskOutput:
    if isinstance(value, ExecutorTaskOutput):
        output = value
    elif isinstance(value, dict):
        structured_response = value.get("structured_response")
        if structured_response is not None:
            return _coerce_executor_output(structured_response, task)
        output = ExecutorTaskOutput.model_validate(value)
    elif isinstance(value, str):
        output = ExecutorTaskOutput(task_id=task.task_id, status="success", summary=value)
    else:
        output = ExecutorTaskOutput(task_id=task.task_id, status="success", raw_output=value)
    if not output.task_id:
        output.task_id = task.task_id
    return output


def _review_component_plan(
    plan: ComponentImplementationPlan,
    target_component: str,
    module_design: ModuleDesignOutput,
) -> PlanningReviewOutput:
    required_changes: list[str] = []
    task_by_id = {task.task_id: task for task in plan.tasks}
    module_names = [module.name for module in module_design.modules]

    if plan.component_name != target_component:
        required_changes.append("Plan component_name must match the requested target_component.")

    for module_name in module_names:
        matching_node = next((node for node in plan.module_nodes if node.module_name == module_name), None)
        if matching_node is None:
            required_changes.append(f"Missing module plan node for {module_name}.")
            continue

        implementation_task = task_by_id.get(matching_node.implementation_task_id)
        test_task = task_by_id.get(matching_node.test_task_id)
        execute_task = task_by_id.get(matching_node.execute_test_task_id)

        if implementation_task is None or implementation_task.kind != "create_file":
            required_changes.append(f"{module_name} must have a create_file implementation task.")
        if test_task is None or test_task.kind != "create_test_file":
            required_changes.append(f"{module_name} must have a create_test_file task.")
        if execute_task is None or execute_task.kind != "execute_test":
            required_changes.append(f"{module_name} must have an execute_test task.")
        elif execute_task.test_target_task_id != matching_node.test_task_id:
            required_changes.append(f"{module_name} execute_test must target its test task.")

    if plan.module_nodes:
        integration_task = task_by_id.get(plan.integration_test_task_id or "")
        execute_integration_task = task_by_id.get(plan.execute_integration_test_task_id or "")
        if integration_task is None or integration_task.kind != "create_integration_test":
            required_changes.append("Plan must include a create_integration_test task.")
        if execute_integration_task is None or execute_integration_task.kind != "execute_integration_test":
            required_changes.append("Plan must include an execute_integration_test task.")
        elif execute_integration_task.test_target_task_id != plan.integration_test_task_id:
            required_changes.append("execute_integration_test must target the integration test task.")

    if required_changes:
        return PlanningReviewOutput(
            approved=False,
            feedback="Component implementation plan failed deterministic review.",
            required_changes=required_changes,
        )
    return PlanningReviewOutput(
        approved=True,
        feedback="Component implementation plan passed deterministic review.",
    )


def _format_review_failure(review: PlanningReviewOutput) -> str:
    if not review.required_changes:
        return review.feedback
    return f"{review.feedback} Required changes: {'; '.join(review.required_changes)}"


def _format_planning_revision_feedback(
    plan: ComponentImplementationPlan,
    review: PlanningReviewOutput,
    human_feedback: str = "",
) -> str:
    planned_modules = ", ".join(node.module_name for node in plan.module_nodes) or "None"
    task_kind_counts: dict[str, int] = {}
    for task in plan.tasks:
        task_kind_counts[task.kind] = task_kind_counts.get(task.kind, 0) + 1
    task_count_text = ", ".join(
        f"{kind}={count}" for kind, count in sorted(task_kind_counts.items())
    ) or "None"

    feedback = (
        "Previous implementation plan failed deterministic review.\n"
        f"{_format_review_failure(review)}\n\n"
        "Previous plan diagnostics:\n"
        f"- Planned module nodes: {planned_modules}\n"
        f"- Integration test task id: {plan.integration_test_task_id or 'None'}\n"
        f"- Execute integration test task id: {plan.execute_integration_test_task_id or 'None'}\n"
        f"- Task kind counts: {task_count_text}\n\n"
        "Revise the plan so every module in module_design.modules has a ModulePlanNode "
        "with create_file, create_test_file, and execute_test tasks, then add the final "
        "create_integration_test and execute_integration_test tasks."
    )
    if human_feedback:
        feedback = f"{feedback}\n\nHuman feedback:\n{human_feedback}"
    return feedback


def _append_planning_feedback(existing_feedback: str, new_feedback: str) -> str:
    if not existing_feedback:
        return new_feedback
    if not new_feedback:
        return existing_feedback
    return f"{existing_feedback}\n\n{new_feedback}"


def _build_task_progress(plan: ComponentImplementationPlan) -> list[PlanningTaskProgress]:
    return [
        PlanningTaskProgress(
            task_id=task.task_id,
            kind=task.kind,
            relative_path=task.relative_path,
        )
        for task in plan.tasks
    ]


def _require_current_task(state: ImplementationState) -> PlanningTask:
    task = _find_task(state["implementation_plan"], state.get("current_task_id"))
    if task is None:
        raise ValueError("Current planning task is not set")
    return task


def _find_task(plan: ComponentImplementationPlan, task_id: str | None) -> PlanningTask | None:
    if task_id is None:
        return None
    for task in plan.tasks:
        if task.task_id == task_id:
            return task
    return None


def _next_ready_task(state: ImplementationState) -> PlanningTask | None:
    forced_task = _find_task(state["implementation_plan"], state.get("current_task_id"))
    if forced_task and _task_status(state, forced_task.task_id) in {"pending", "failed"}:
        return forced_task

    completed_ids = {
        progress.task_id
        for progress in state.get("task_progress", [])
        if progress.status == "completed"
    }
    for task in state["implementation_plan"].tasks:
        if _task_status(state, task.task_id) == "completed":
            continue
        if all(dependency in completed_ids for dependency in task.depends_on):
            return task
    return None


def _route_by_current_task(state: ImplementationState) -> str:
    if state.get("status") == "failed":
        return "failed"
    if _all_tasks_completed(state):
        return "complete"
    task = _find_task(state["implementation_plan"], state.get("current_task_id"))
    if task is None:
        return "failed"
    if task.kind in WRITER_TASK_KINDS:
        return "writer"
    if task.kind in EXECUTOR_TASK_KINDS:
        return "executor"
    return "failed"


def _task_status(state: ImplementationState, task_id: str) -> TaskProgressStatus:
    for progress in state.get("task_progress", []):
        if progress.task_id == task_id:
            return progress.status
    return "pending"


def _update_task_status(
    task_progress: list[PlanningTaskProgress],
    task_id: str,
    status: TaskProgressStatus,
) -> list[PlanningTaskProgress]:
    return [
        progress.model_copy(update={"status": status}) if progress.task_id == task_id else progress
        for progress in task_progress
    ]


def _repair_task_id_for_execution_failure(
    plan: ComponentImplementationPlan,
    execution_task: PlanningTask,
) -> str | None:
    if execution_task.test_target_task_id:
        return execution_task.test_target_task_id
    for task_id in reversed(execution_task.depends_on):
        if _find_task(plan, task_id) is not None:
            return task_id
    return None


def _all_tasks_completed(state: ImplementationState) -> bool:
    task_progress = state.get("task_progress", [])
    return bool(task_progress) and all(progress.status == "completed" for progress in task_progress)


def _inspect_repository_root() -> Any:
    return inspect_repository.invoke(
        {
            "relative_path": ".",
            "max_depth": 5,
            "max_entries": 300,
            "extra_ignored_names": [".git", ".venv", "__pycache__", "node_modules"],
        }
    )


def _execute_repository_command(command: list[str], timeout_s: int) -> Any:
    return run_repository_command.invoke({"command": command, "timeout_s": timeout_s})


def _write_repository_file(relative_path: str, content: str) -> FileWriteResult:
    try:
        repo_root = _resolve_repository_root()
        file_path = (repo_root / relative_path).expanduser().resolve()
        if file_path != repo_root and repo_root not in file_path.parents:
            return FileWriteResult(
                relative_path=relative_path,
                status="error",
                message="Implementation file path must stay inside the configured repository.",
            )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return FileWriteResult(
            relative_path=relative_path,
            status="success",
            message="File written successfully.",
        )
    except Exception as error:
        return FileWriteResult(
            relative_path=relative_path,
            status="error",
            message=f"Failed to write file: {error}",
        )


def _resolve_repository_root() -> Path:
    if settings.WORKPLACE_FOLDER is None:
        raise ValueError("WORKPLACE_FOLDER must be set in the .env file.")
    if settings.REPO_NAME is None:
        raise ValueError("REPO_NAME must be set in the .env file.")
    workspace_root = Path(settings.WORKPLACE_FOLDER).expanduser().resolve()
    repo_root = (workspace_root / settings.REPO_NAME).expanduser().resolve()
    if repo_root != workspace_root and workspace_root not in repo_root.parents:
        raise ValueError("Repository path must stay inside the configured workspace.")
    return repo_root


def _extract_repository_tree(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("tree") or value.get("message") or value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return _extract_repository_tree(parsed)
    return str(value)


def _next_pending_file(progress_items: list[ImplementationFileProgress]) -> ImplementationFileProgress | None:
    for progress in progress_items:
        if progress.status == "pending":
            return progress
    return None


def _find_file_progress(
    progress_items: list[ImplementationFileProgress],
    file_id: str | None,
) -> ImplementationFileProgress | None:
    if file_id is None:
        return None
    for progress in progress_items:
        if progress.file_id == file_id:
            return progress
    return None


def _update_file_status(
    progress_items: list[ImplementationFileProgress],
    file_id: str | None,
    status: ImplementationFileStatus,
    repair_attempts: int | None = None,
) -> list[ImplementationFileProgress]:
    updated = []
    for progress in progress_items:
        if progress.file_id != file_id:
            updated.append(progress)
            continue
        update_values: dict[str, Any] = {"status": status}
        if repair_attempts is not None:
            update_values["repair_attempts"] = repair_attempts
        updated.append(progress.model_copy(update=update_values))
    return updated


def _require_current_file(state: ImplementationExecutionState) -> ImplementationFilePlan:
    current_file_id = state.get("current_file_id")
    for file_plan in state["planning_output"].implementation_plan.files:
        if file_plan.file_id == current_file_id:
            return file_plan
    raise ValueError(f"Current implementation file is not set or unknown: {current_file_id}")


def _related_test_nodes(
    file_plan: ImplementationFilePlan,
    test_nodes: list[TestPlanNode],
) -> list[TestPlanNode]:
    file_modules = set(file_plan.modules)
    return [
        node
        for node in test_nodes
        if file_modules.intersection(node.target_modules)
    ]


def _targeted_test_command(
    file_plan: ImplementationFilePlan,
    planning_output: PlanningStageOutput,
) -> list[str] | None:
    path = file_plan.relative_path
    path_parts = Path(path).parts
    file_name = Path(path).name
    if not (path_parts and path_parts[0] == "tests") and "test" not in file_name:
        return None
    if planning_output.test_plan.testing_framework.lower() == "pytest":
        return ["pytest", path]
    for command in _normalise_commands(planning_output.test_plan.commands):
        if "pytest" in command:
            return [*command, path] if path not in command else command
    return ["pytest", path]


def _final_verification_commands(
    planning_output: PlanningStageOutput,
    environment_setup: EnvironmentSetupPlan | None,
) -> list[list[str]]:
    commands = _normalise_commands(planning_output.test_plan.commands)
    if commands:
        return commands
    if environment_setup and environment_setup.test_commands:
        return environment_setup.test_commands
    return [["pytest"]]


def _normalise_commands(commands: list[Any]) -> list[list[str]]:
    normalised: list[list[str]] = []
    for command in commands:
        if isinstance(command, str):
            normalised.append(command.split())
        elif isinstance(command, list):
            normalised.append([str(part) for part in command])
    return normalised


def _coerce_test_execution_result(value: Any, command: list[str]) -> TestExecutionResult:
    if isinstance(value, TestExecutionResult):
        return value
    if isinstance(value, dict):
        return TestExecutionResult(
            command=command,
            status=str(value.get("status") or "unknown"),
            exit_code=value.get("exit_code"),
            stdout=str(value.get("stdout") or ""),
            stderr=str(value.get("stderr") or ""),
            message=str(value.get("message") or ""),
            summary=str(value.get("summary") or value.get("message") or ""),
        )
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return TestExecutionResult(
                command=command,
                status="success",
                stdout=value,
                summary=value[:200],
            )
        return _coerce_test_execution_result(parsed, command)
    return TestExecutionResult(
        command=command,
        status="success",
        summary=str(value),
    )


def _build_implementation_stage_output(
    state: ImplementationExecutionState,
    approved: bool,
) -> ImplementationStageOutput:
    status = "complete" if approved else "failed"
    execution = ImplementationExecutionResult(
        file_writes=state.get("file_writes", []),
        test_executions=state.get("test_executions", []),
        status=status,
        summary=(
            "Implementation execution completed successfully."
            if approved
            else state.get("failure_reason", "Implementation execution failed.")
        ),
    )
    return ImplementationStageOutput(
        execution=execution,
        approved=approved,
        failure_reason="" if approved else state.get("failure_reason", ""),
    )


def _format_exception(error: Exception) -> str:
    message = str(error).strip()
    if not message:
        message = error.__class__.__name__
    return f"{error.__class__.__name__}: {message}"


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"\nError: {exc}") from None

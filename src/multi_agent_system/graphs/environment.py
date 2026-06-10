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
    DirectorySpec,
    EnvironmentSetupExecution,
    EnvironmentSetupPlan,
    EnvironmentSetupStageOutput,
    ModuleDesignOutput,
    RepositoryStructure,
    RequirementsStageOutput,
    SetupExecutionResult,
    SetupFileSpec,
)
from src.multi_agent_system.store import ProjectStoreRepository
from src.settings import settings
from src.tools.docker_code_execution.repo_tool import run_repository_command
from src.tools.file_system_tools.inspect_repository import inspect_repository


EnvironmentStatus = Literal[
    "complete",
    "needs_revision",
    "waiting_for_approval",
    "aborted",
    "failed",
]
ApprovalDecision = Literal["Y", "N", "A"]
AgentRunner = Callable[[Any, BaseModel], Any]
RepositoryInspector = Callable[[], Any]
CommandExecutor = Callable[[list[str], int], Any]
ApprovalCallback = Callable[[EnvironmentSetupStageOutput, str], ApprovalDecision]
FailureAdviceCallback = Callable[[SetupExecutionResult], str]
SetupDirectoryCreator = Callable[[DirectorySpec], SetupExecutionResult]
SetupFileWriter = Callable[[SetupFileSpec], SetupExecutionResult]
ModelT = TypeVar("ModelT", bound=BaseModel)


DEFAULT_REQUIRED_REPOSITORY_ENTRIES = ("src", "tests", ".env", ".gitignore")
DEFAULT_COMMAND_TIMEOUT_S = 120


ENVIRONMENT_AGENT_PROMPT = """
You are the Environment Agent that either updates or creates a environment setup plan

Create a new or improve an environment setup plan given to you for the target repository. Use the repository tree,
approved requirements, high-level design, component decompositions, module designs, previous
plan feedback, and any failed command output provided in the input.

Your output must be an EnvironmentSetupStageOutput. Keep approved=false; user approval happens
in the graph approval node.

Planning rules:
- Analyse the High level design and low level design artifacts before choosing commands.
- Use only the repository tree and the provided design/store artifacts for planning.
- You do not have tools. Do not ask to inspect files. The repository_tree input is the only repository inspection context.
- Do not create application source files or test source files in setup_files.
- Only use setup_files for environment/configuration files, dependency files, or ignore files.
- The graph creates directories from repository_structure before writing setup_files.
- The graph writes setup_files before running setup_commands, build_commands, or test_commands.
- If a command requires a folder, include it in repository_structure.
- If a command requires a file or folder, include the required file in setup_files or create it with an earlier command.
- If setup_commands reference requirements.txt, pom.xml, package.json, or another dependency file, that file must exist first.
- Prefer the default Python repository structure unless project evidence says otherwise:
  src/, tests/, .env, .gitignore.
- Keep setup minimal and safe.
- Do not include destructive commands.
- Prefer command arrays, for example ["python", "-m", "venv", ".venv"].
- Commands run directly in Docker, not through an interactive shell. Do not rely on shell builtins like source.
- Virtual environments should be used by calling their executables directly, for example [".venv/bin/python", "-m", "pytest"] or [".venv/bin/pip", "install", "-r", "requirements.txt"].
- Activation commands such as ["source", ".venv/bin/activate"] will fail because source is a shell builtin and activation would not persist across command executions.
- Prefer project-local wrappers when present in the repository tree, for example ["./mvnw", "test"] or ["./gradlew", "test"].
- For Python projects, commonly useful commands include creating a virtual environment,
  installing dependencies from requirements.txt, and running pytest when tests exist.
- For Java Maven projects, use ["./mvnw", "test"] only when mvnw is visible in the repository tree; otherwise avoid assuming mvn exists.
- For Node projects, only use npm commands when package.json or design evidence supports Node.
- If a previous command failed, revise the plan from that stage and avoid repeating the same
  failing command unless the plan explicitly fixes its cause first.
""".strip()


class EnvironmentAgentInput(BaseModel):
    project_prompt: str
    requirements: RequirementsStageOutput | None = None
    component_extraction: ComponentExtractionOutput | None = None
    component_decompositions: dict[str, ComponentDecompositionOutput] = Field(default_factory=dict)
    module_designs: dict[str, ModuleDesignOutput] = Field(default_factory=dict)
    repository_tree: str = ""
    previous_stage_output: EnvironmentSetupStageOutput | None = None
    evaluation_feedback: str = ""
    failed_command_result: SetupExecutionResult | None = None


class EnvironmentEvaluation(BaseModel):
    approved: bool = True
    feedback: str = ""
    required_changes: list[str] = Field(default_factory=list)


class EnvironmentState(TypedDict, total=False):
    project_id: str
    project_prompt: str
    requirements: RequirementsStageOutput | None
    component_extraction: ComponentExtractionOutput | None
    component_decompositions: dict[str, ComponentDecompositionOutput]
    module_designs: dict[str, ModuleDesignOutput]
    repository_tree: str
    stage_output: EnvironmentSetupStageOutput
    previous_stage_output: EnvironmentSetupStageOutput | None
    evaluation_output: EnvironmentEvaluation
    evaluation_feedback: str
    approval_decision: ApprovalDecision
    execution_result: EnvironmentSetupExecution
    failed_command_result: SetupExecutionResult | None
    status: EnvironmentStatus
    failure_reason: str
    revision_count: int


EnvironmentEvaluator = Callable[[EnvironmentState], EnvironmentEvaluation]


class EnvironmentRunResult(BaseModel):
    status: EnvironmentStatus
    stage_output: EnvironmentSetupStageOutput | None = None
    execution_result: EnvironmentSetupExecution | None = None
    failed_command_result: SetupExecutionResult | None = None
    repository_tree: str = ""
    evaluation_feedback: str = ""
    failure_reason: str = ""
    revision_count: int = 0


def default_environment_approval(
    stage_output: EnvironmentSetupStageOutput,
    evaluation_feedback: str,
) -> ApprovalDecision:
    print("\nApproval required for environment setup plan")
    print("Repository structure:")
    print(_format_repository_structure(stage_output.repository_structure))
    print("\nSetup files:")
    if stage_output.environment_setup.setup_files:
        for setup_file in stage_output.environment_setup.setup_files:
            print(f"- {setup_file.path}: {setup_file.description}")
    else:
        print("- None")
    print("\nSetup commands:")
    _print_commands(stage_output.environment_setup.setup_commands)
    print("\nBuild commands:")
    _print_commands(stage_output.environment_setup.build_commands)
    print("\nTest commands:")
    _print_commands(stage_output.environment_setup.test_commands)
    if stage_output.notes:
        print("\nNotes:")
        for note in stage_output.notes:
            print(f"- {note}")
    if evaluation_feedback:
        print("\nEvaluation:")
        print(evaluation_feedback)

    while True:
        choice = input("Approve environment setup? [Y=yes, N=re-evaluate, A=abort]: ").strip().lower()
        if choice in {"y", "yes"}:
            return "Y"
        if choice in {"n", "no"}:
            return "N"
        if choice in {"a", "abort"}:
            return "A"
        print("Please enter Y, N, or A.")


def default_failure_advice(result: SetupExecutionResult) -> str:
    print("\nEnvironment setup command failed")
    print(f"Step: {result.step}")
    print(f"Command: {' '.join(result.command)}")
    print(f"Exit code: {result.exit_code}")
    if result.message:
        print(f"Message: {result.message}")
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    return input("Advice for the Environment Agent before replanning (optional): ").strip()


class EnvironmentStageGraph:
    """LangGraph for repository environment planning, approval, and command execution."""

    def __init__(
        self,
        environment_agent: Any | None = None,
        environment_runner: AgentRunner | None = None,
        repository_inspector: RepositoryInspector | None = None,
        environment_evaluator: EnvironmentEvaluator | None = None,
        command_executor: CommandExecutor | None = None,
        setup_directory_creator: SetupDirectoryCreator | None = None,
        setup_file_writer: SetupFileWriter | None = None,
        approval_callback: ApprovalCallback | None = None,
        failure_advice_callback: FailureAdviceCallback | None = None,
        store_repository: ProjectStoreRepository | None = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_revisions: int = 5,
        command_timeout_s: int = DEFAULT_COMMAND_TIMEOUT_S,
    ) -> None:
        self.environment_agent = environment_agent
        self.environment_runner = environment_runner or _run_structured_agent
        self.repository_inspector = repository_inspector or _inspect_repository_root
        self.environment_evaluator = environment_evaluator or _evaluate_environment_state
        self.command_executor = command_executor or _execute_repository_command
        self.setup_directory_creator = setup_directory_creator or _create_repository_setup_directory
        self.setup_file_writer = setup_file_writer or _write_repository_setup_file
        self.approval_callback = approval_callback or default_environment_approval
        self.failure_advice_callback = failure_advice_callback or default_failure_advice
        self.store_repository = store_repository
        self.model_name = model_name
        self.temperature = temperature
        self.max_revisions = max_revisions
        self.command_timeout_s = command_timeout_s
        self.graph = self._build_environment_graph()

    def run(
        self,
        project_prompt: str,
        project_id: str | None = None,
        requirements: RequirementsStageOutput | None = None,
        component_extraction: ComponentExtractionOutput | None = None,
        component_decompositions: dict[str, ComponentDecompositionOutput] | None = None,
        module_designs: dict[str, ModuleDesignOutput] | None = None,
        previous_stage_output: EnvironmentSetupStageOutput | None = None,
    ) -> EnvironmentRunResult:
        initial_state: EnvironmentState = {
            "project_prompt": project_prompt,
            "requirements": requirements,
            "component_extraction": component_extraction,
            "component_decompositions": component_decompositions or {},
            "module_designs": module_designs or {},
            "previous_stage_output": previous_stage_output,
            "revision_count": 0,
        }
        if project_id is not None:
            initial_state["project_id"] = project_id
        final_state = self.graph.invoke(initial_state)
        return self._to_result(final_state)

    def _build_environment_graph(self):
        graph_builder = StateGraph(EnvironmentState)
        graph_builder.add_node("inspect_repository_structure", self.inspect_repository_structure)
        graph_builder.add_node("environment_agent_plan", self.environment_agent_plan)
        graph_builder.add_node("self_evaluate_plan", self.self_evaluate_plan)
        graph_builder.add_node("user_approval", self.user_approval)
        graph_builder.add_node("execute_setup_commands", self.execute_setup_commands)
        graph_builder.add_node("complete", self.complete)
        graph_builder.add_node("aborted", self.aborted)
        graph_builder.add_node("failed", self.failed)

        graph_builder.add_edge(START, "inspect_repository_structure")
        graph_builder.add_edge("inspect_repository_structure", "environment_agent_plan")
        graph_builder.add_conditional_edges(
            "environment_agent_plan",
            self.route_after_environment_agent_plan,
            {
                "self_evaluate_plan": "self_evaluate_plan",
                "user_approval": "user_approval",
                "failed": "failed",
            },
        )
        graph_builder.add_conditional_edges(
            "self_evaluate_plan",
            self.route_after_self_evaluation,
            {
                "user_approval": "user_approval",
                "environment_agent_plan": "environment_agent_plan",
                "failed": "failed",
            },
        )
        graph_builder.add_conditional_edges(
            "user_approval",
            self.route_after_user_approval,
            {
                "execute_setup_commands": "execute_setup_commands",
                "environment_agent_plan": "environment_agent_plan",
                "aborted": "aborted",
                "failed": "failed",
            },
        )
        graph_builder.add_conditional_edges(
            "execute_setup_commands",
            self.route_after_execution,
            {
                "complete": "complete",
                "environment_agent_plan": "environment_agent_plan",
            },
        )
        graph_builder.add_edge("complete", END)
        graph_builder.add_edge("aborted", END)
        graph_builder.add_edge("failed", END)
        return graph_builder.compile()

    def inspect_repository_structure(self, state: EnvironmentState) -> EnvironmentState:
        print("Node: inspect_repository_structure")
        raw_output = self.repository_inspector()
        repository_tree = _extract_repository_tree(raw_output)
        return {
            **state,
            "repository_tree": repository_tree,
        }

    def environment_agent_plan(self, state: EnvironmentState) -> EnvironmentState:
        print("Node: environment_agent_plan")

        revision_count = state.get("revision_count", 0) + 1
        if revision_count > self.max_revisions:
            return {
                **state,
                "revision_count": revision_count,
                "status": "failed",
                "failure_reason": f"Maximum environment setup revisions exceeded: {self.max_revisions}",
            }

        agent_input = EnvironmentAgentInput(
            project_prompt=state["project_prompt"],
            requirements=state.get("requirements"),
            component_extraction=state.get("component_extraction"),
            component_decompositions=state.get("component_decompositions", {}),
            module_designs=state.get("module_designs", {}),
            repository_tree=state.get("repository_tree", ""),
            previous_stage_output=state.get("stage_output") or state.get("previous_stage_output"),
            evaluation_feedback=state.get("evaluation_feedback", ""),
            failed_command_result=state.get("failed_command_result"),
        )
        raw_output = self.environment_runner(
            self._get_environment_agent(),
            agent_input,
        )
        stage_output = _coerce_model(raw_output, EnvironmentSetupStageOutput)
        stage_output.approved = False
        return {
            **state,
            "stage_output": stage_output,
            "previous_stage_output": stage_output,
            "revision_count": revision_count,
        }

    def self_evaluate_plan(self, state: EnvironmentState) -> EnvironmentState:
        print("Node: self_evaluate_plan")
        evaluation = self.environment_evaluator(state)
        if evaluation.approved:
            return {
                **state,
                "evaluation_output": evaluation,
                "evaluation_feedback": evaluation.feedback,
            }

        feedback = _format_evaluation_feedback(evaluation)
        return {
            **state,
            "evaluation_output": evaluation,
            "evaluation_feedback": feedback,
        }

    @staticmethod
    def route_after_environment_agent_plan(state: EnvironmentState) -> str:
        if state.get("status") == "failed":
            return "failed"
        if state.get("revision_count", 0) <= 1:
            return "self_evaluate_plan"
        return "user_approval"

    def user_approval(self, state: EnvironmentState) -> EnvironmentState:
        print("Node: user_approval")
        stage_output = state["stage_output"]
        decision = self.approval_callback(stage_output, state.get("evaluation_feedback", ""))
        normalized_decision = _normalize_approval_decision(decision)

        if normalized_decision == "Y":
            stage_output = stage_output.model_copy(update={"approved": True, "approval_feedback": ""})
            self._save_environment_setup_plan(state, stage_output)
            return {
                **state,
                "stage_output": stage_output,
                "approval_decision": "Y",
                "status": "waiting_for_approval",
            }

        if normalized_decision == "N":
            feedback = _append_feedback(
                state.get("evaluation_feedback", ""),
                "User requested environment setup plan re-evaluation.",
            )
            stage_output = stage_output.model_copy(
                update={
                    "approved": False,
                    "approval_feedback": "User requested re-evaluation.",
                }
            )
            return {
                **state,
                "stage_output": stage_output,
                "approval_decision": "N",
                "evaluation_feedback": feedback,
                "status": "needs_revision",
            }

        return {
            **state,
            "approval_decision": "A",
            "status": "aborted",
        }

    def execute_setup_commands(self, state: EnvironmentState) -> EnvironmentState:
        print("Node: execute_setup_commands")
        stage_output = state["stage_output"]
        results: list[SetupExecutionResult] = []
        for directory in stage_output.repository_structure.directories:
            result = self.setup_directory_creator(directory)
            results.append(result)
            if result.status != "success":
                execution = EnvironmentSetupExecution(
                    results=results,
                    status="failed",
                    summary=f"Failed to create setup directory: {directory.name}",
                )
                self._save_environment_setup_execution(state, execution)
                user_advice = self.failure_advice_callback(result)
                feedback = _append_feedback(
                    state.get("evaluation_feedback", ""),
                    _format_failed_command_feedback(result, user_advice),
                )
                return {
                    **state,
                    "execution_result": execution,
                    "failed_command_result": result,
                    "evaluation_feedback": feedback,
                    "status": "needs_revision",
                }

        for setup_file in stage_output.environment_setup.setup_files:
            result = self.setup_file_writer(setup_file)
            results.append(result)
            if result.status != "success":
                execution = EnvironmentSetupExecution(
                    results=results,
                    status="failed",
                    summary=f"Failed to write setup file: {setup_file.path}",
                )
                self._save_environment_setup_execution(state, execution)
                user_advice = self.failure_advice_callback(result)
                feedback = _append_feedback(
                    state.get("evaluation_feedback", ""),
                    _format_failed_command_feedback(result, user_advice),
                )
                return {
                    **state,
                    "execution_result": execution,
                    "failed_command_result": result,
                    "evaluation_feedback": feedback,
                    "status": "needs_revision",
                }

        for step_name, commands in _iter_commands(stage_output.environment_setup):
            for command in commands:
                result = self._execute_command(step_name, command)
                results.append(result)
                if result.status != "success":
                    execution = EnvironmentSetupExecution(
                        results=results,
                        status="failed",
                        summary=f"Command failed during {step_name}: {' '.join(command)}",
                    )
                    self._save_environment_setup_execution(state, execution)
                    user_advice = self.failure_advice_callback(result)
                    feedback = _append_feedback(
                        state.get("evaluation_feedback", ""),
                        _format_failed_command_feedback(result, user_advice),
                    )
                    return {
                        **state,
                        "execution_result": execution,
                        "failed_command_result": result,
                        "evaluation_feedback": feedback,
                        "status": "needs_revision",
                    }

        execution = EnvironmentSetupExecution(
            results=results,
            status="complete",
            summary="Environment setup commands completed successfully.",
        )
        self._save_environment_setup_execution(state, execution)
        return {
            **state,
            "execution_result": execution,
            "failed_command_result": None,
            "status": "complete",
        }

    @staticmethod
    def route_after_self_evaluation(state: EnvironmentState) -> str:
        if state.get("status") == "failed":
            return "failed"
        evaluation = state.get("evaluation_output") or _evaluate_environment_state(state)
        if evaluation.approved:
            return "user_approval"
        return "environment_agent_plan"

    @staticmethod
    def route_after_user_approval(state: EnvironmentState) -> str:
        decision = state.get("approval_decision")
        if decision == "Y":
            return "execute_setup_commands"
        if decision == "N":
            return "environment_agent_plan"
        if decision == "A":
            return "aborted"
        return "failed"

    @staticmethod
    def route_after_execution(state: EnvironmentState) -> str:
        if state.get("failed_command_result") is not None:
            return "environment_agent_plan"
        return "complete"

    @staticmethod
    def complete(state: EnvironmentState) -> EnvironmentState:
        print("Node: complete")
        return {
            **state,
            "status": "complete",
        }

    @staticmethod
    def aborted(state: EnvironmentState) -> EnvironmentState:
        print("Node: aborted")
        return {
            **state,
            "status": "aborted",
            "failure_reason": "Environment setup aborted by user.",
        }

    @staticmethod
    def failed(state: EnvironmentState) -> EnvironmentState:
        print("Node: failed")
        return {
            **state,
            "status": "failed",
            "failure_reason": state.get("failure_reason", "Environment setup failed."),
        }

    def _execute_command(self, step_name: str, command: list[str]) -> SetupExecutionResult:
        raw_output = self.command_executor(command, self.command_timeout_s)
        return _coerce_execution_result(
            raw_output=raw_output,
            step_name=step_name,
            command=command,
        )

    def _get_environment_agent(self) -> Any:
        if self.environment_agent is None:
            self.environment_agent = AgentFactory.build_agent(
                prompt=ENVIRONMENT_AGENT_PROMPT,
                tools=[],
                temperature=self.temperature,
                model_name=self.model_name,
                response_format=EnvironmentSetupStageOutput,
            )
        return self.environment_agent

    def _save_environment_setup_plan(
        self,
        state: EnvironmentState,
        stage_output: EnvironmentSetupStageOutput,
    ) -> None:
        if self.store_repository is None or not state.get("project_id"):
            return
        self.store_repository.save_environment_setup_plan(state["project_id"], stage_output)

    def _save_environment_setup_execution(
        self,
        state: EnvironmentState,
        execution: EnvironmentSetupExecution,
    ) -> None:
        if self.store_repository is None or not state.get("project_id"):
            return
        self.store_repository.save_environment_setup_execution(state["project_id"], execution)

    @staticmethod
    def _to_result(state: EnvironmentState) -> EnvironmentRunResult:
        return EnvironmentRunResult(
            status=state.get("status", "failed"),
            stage_output=state.get("stage_output"),
            execution_result=state.get("execution_result"),
            failed_command_result=state.get("failed_command_result"),
            repository_tree=state.get("repository_tree", ""),
            evaluation_feedback=state.get("evaluation_feedback", ""),
            failure_reason=state.get("failure_reason", ""),
            revision_count=state.get("revision_count", 0),
        )


def build_environment_graph(**kwargs: Any) -> EnvironmentStageGraph:
    return EnvironmentStageGraph(**kwargs)


def _inspect_repository_root() -> Any:
    return inspect_repository.invoke(
        {
            "relative_path": ".",
            "max_depth": 4,
            "max_entries": 200,
            "extra_ignored_names": [".git", ".venv", "__pycache__", "node_modules"],
        }
    )


def _execute_repository_command(command: list[str], timeout_s: int) -> Any:
    return run_repository_command.invoke({"command": command, "timeout_s": timeout_s})


def _create_repository_setup_directory(directory: DirectorySpec) -> SetupExecutionResult:
    try:
        repo_path = _resolve_repository_path()
        directory_path = _resolve_directory_spec_path(repo_path, directory)
        if directory_path != repo_path and repo_path not in directory_path.parents:
            return SetupExecutionResult(
                step="setup_directories",
                tool_name="create_setup_directory",
                path=directory.name,
                status="error",
                message="Setup directory path must stay inside the configured repository.",
            )

        directory_path.mkdir(parents=True, exist_ok=True)
        return SetupExecutionResult(
            step="setup_directories",
            tool_name="create_setup_directory",
            path=str(directory_path.relative_to(repo_path)),
            status="success",
            exit_code=0,
            message="Setup directory created successfully.",
        )
    except Exception as error:
        return SetupExecutionResult(
            step="setup_directories",
            tool_name="create_setup_directory",
            path=directory.name,
            status="error",
            message=f"Failed to create setup directory: {error}",
        )


def _write_repository_setup_file(setup_file: SetupFileSpec) -> SetupExecutionResult:
    try:
        repo_path = _resolve_repository_path()
        file_path = (repo_path / setup_file.path).expanduser().resolve()
        if file_path != repo_path and repo_path not in file_path.parents:
            return SetupExecutionResult(
                step="setup_files",
                tool_name="write_setup_file",
                path=setup_file.path,
                status="error",
                message="Setup file path must stay inside the configured repository.",
            )

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(setup_file.content, encoding="utf-8")
        return SetupExecutionResult(
            step="setup_files",
            tool_name="write_setup_file",
            path=setup_file.path,
            status="success",
            exit_code=0,
            message="Setup file written successfully.",
        )
    except Exception as error:
        return SetupExecutionResult(
            step="setup_files",
            tool_name="write_setup_file",
            path=setup_file.path,
            status="error",
            message=f"Failed to write setup file: {error}",
        )


def _resolve_repository_path() -> Path:
    if settings.WORKPLACE_FOLDER is None:
        raise ValueError("WORKPLACE_FOLDER must be set in the .env file.")
    if settings.REPO_NAME is None:
        raise ValueError("REPO_NAME must be set in the .env file.")

    workspace_root = Path(settings.WORKPLACE_FOLDER).expanduser().resolve()
    repo_path = (workspace_root / settings.REPO_NAME).expanduser().resolve()
    if repo_path != workspace_root and workspace_root not in repo_path.parents:
        raise ValueError("Repository path must stay inside the configured workplace folder.")
    return repo_path


def _resolve_directory_spec_path(repo_path: Path, directory: DirectorySpec) -> Path:
    if directory.parent:
        return (repo_path / directory.parent / directory.name).expanduser().resolve()
    return (repo_path / directory.name).expanduser().resolve()


def _run_structured_agent(agent: Any, agent_input: BaseModel) -> Any:
    payload = agent_input.model_dump(mode="json")
    return agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=json.dumps(payload, indent=2),
                )
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


def _evaluate_environment_state(state: EnvironmentState) -> EnvironmentEvaluation:
    return _evaluate_environment_plan(state["stage_output"])


def _evaluate_environment_plan(stage_output: EnvironmentSetupStageOutput) -> EnvironmentEvaluation:
    required_changes: list[str] = []
    repository_entries = _repository_entry_names(stage_output.repository_structure)
    for entry in DEFAULT_REQUIRED_REPOSITORY_ENTRIES:
        if entry not in repository_entries:
            required_changes.append(
                f"Repository structure should include {entry} unless project-specific evidence says otherwise."
            )

    for command in _all_commands(stage_output.environment_setup):
        if _is_destructive_command(command):
            required_changes.append(
                f"Remove destructive command from setup plan: {' '.join(command)}"
            )

    if required_changes:
        return EnvironmentEvaluation(
            approved=False,
            feedback="Environment setup plan needs revision.",
            required_changes=required_changes,
        )

    return EnvironmentEvaluation(
        approved=True,
        feedback="Environment setup plan passed self-evaluation.",
    )


def _repository_entry_names(repository_structure: RepositoryStructure) -> set[str]:
    names: set[str] = set()
    for directory in repository_structure.directories:
        if directory.name:
            names.add(directory.name)
        for file_spec in directory.files:
            if file_spec.name:
                names.add(file_spec.name)
    return names


def _is_destructive_command(command: list[str]) -> bool:
    if not command:
        return False
    destructive_tokens = {"rm", "rmdir", "del", "erase"}
    destructive_flags = {"--force", "-rf", "-fr"}
    first = command[0].lower()
    return first in destructive_tokens or any(part.lower() in destructive_flags for part in command)


def _format_evaluation_feedback(evaluation: EnvironmentEvaluation) -> str:
    if evaluation.approved:
        return evaluation.feedback
    changes = "\n- ".join(evaluation.required_changes)
    return f"{evaluation.feedback}\nRequired changes:\n- {changes}"


def _iter_commands(environment_setup: EnvironmentSetupPlan) -> list[tuple[str, list[list[str]]]]:
    return [
        ("setup_commands", environment_setup.setup_commands),
        ("build_commands", environment_setup.build_commands),
        ("test_commands", environment_setup.test_commands),
    ]


def _all_commands(environment_setup: EnvironmentSetupPlan) -> list[list[str]]:
    commands: list[list[str]] = []
    for _, command_group in _iter_commands(environment_setup):
        commands.extend(command_group)
    return commands


def _coerce_execution_result(
    raw_output: Any,
    step_name: str,
    command: list[str],
) -> SetupExecutionResult:
    parsed = _parse_tool_output(raw_output)
    if isinstance(parsed, SetupExecutionResult):
        result = parsed
    elif isinstance(parsed, dict):
        result = SetupExecutionResult.model_validate(
            {
                "step": step_name,
                "tool_name": "run_repository_command",
                "command": command,
                "status": parsed.get("status", "unknown"),
                "exit_code": parsed.get("exit_code"),
                "stdout": parsed.get("stdout", ""),
                "stderr": parsed.get("stderr", ""),
                "message": parsed.get("message", ""),
                "raw_output": raw_output if isinstance(raw_output, str) else json.dumps(parsed),
            }
        )
    else:
        result = SetupExecutionResult(
            step=step_name,
            tool_name="run_repository_command",
            command=command,
            status="unknown",
            message=str(parsed),
            raw_output=str(raw_output),
        )

    if not result.step:
        result.step = step_name
    if not result.tool_name:
        result.tool_name = "run_repository_command"
    if not result.command:
        result.command = command
    return result


def _parse_tool_output(raw_output: Any) -> Any:
    if isinstance(raw_output, str):
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return raw_output
    return raw_output


def _format_failed_command_feedback(result: SetupExecutionResult, user_advice: str = "") -> str:
    feedback = (
        "Environment setup command failed. Revise the plan from this stage.\n"
        f"Step: {result.step}\n"
        f"Command: {' '.join(result.command)}\n"
        f"Exit code: {result.exit_code}\n"
        f"Message: {result.message}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    if user_advice:
        feedback = f"{feedback}\n\nUser advice:\n{user_advice}"
    return feedback


def _normalize_approval_decision(decision: Any) -> ApprovalDecision:
    value = str(decision).strip().lower()
    if value in {"y", "yes"}:
        return "Y"
    if value in {"n", "no"}:
        return "N"
    if value in {"a", "abort"}:
        return "A"
    raise ValueError("Approval decision must be Y, N, or A")


def _append_feedback(existing_feedback: str, new_feedback: str) -> str:
    if not existing_feedback:
        return new_feedback
    if not new_feedback:
        return existing_feedback
    return f"{existing_feedback}\n\n{new_feedback}"


def _format_repository_structure(repository_structure: RepositoryStructure) -> str:
    if not repository_structure.directories:
        return "(empty)"
    lines = []
    for directory in repository_structure.directories:
        lines.append(f"- {directory.name}/")
        for file_spec in directory.files:
            lines.append(f"  - {file_spec.name}")
    return "\n".join(lines)


def _print_commands(commands: list[list[str]]) -> None:
    if not commands:
        print("- None")
        return
    for command in commands:
        print(f"- {' '.join(command)}")


def _load_environment_inputs_from_store(
    project_id: str = "todo_cli",
    store_repository: ProjectStoreRepository | None = None,
) -> dict[str, Any]:
    repository = store_repository or ProjectStoreRepository()
    store = repository.load_project(project_id)
    return {
        "project_id": store.project_id,
        "project_prompt": store.project_prompt,
        "requirements": store.requirements,
        "component_extraction": store.component_extraction,
        "module_designs": store.module_designs,
    }


def _print_loaded_environment_inputs(inputs: dict[str, Any]) -> None:
    requirements = inputs.get("requirements")
    component_extraction = inputs.get("component_extraction")
    module_designs = inputs.get("module_designs", {})
    component_count = 0
    if component_extraction is not None:
        component_count = len(component_extraction.components)
    module_count = sum(len(module_design.modules) for module_design in module_designs.values())

    print("\nLoaded environment graph inputs")
    print(f"Project id: {inputs.get('project_id')}")
    print(f"Project prompt: {inputs.get('project_prompt')}")
    print(f"Requirements loaded: {requirements is not None}")
    print(f"Components extracted: {component_count}")
    print(f"Module designs loaded: {len(module_designs)}")
    print(f"Modules designed: {module_count}")


def main() -> None:
    project_id = input("Project id to test environment graph [todo_cli]: ").strip() or "todo_cli"
    store_repository = ProjectStoreRepository()

    while True:
        inputs = _load_environment_inputs_from_store(
            project_id=project_id,
            store_repository=store_repository,
        )
        _print_loaded_environment_inputs(inputs)

        graph = build_environment_graph(store_repository=store_repository)
        result = graph.run(**inputs)

        print("\nEnvironment graph result")
        print(f"Status: {result.status}")
        print(f"Revision count: {result.revision_count}")
        if result.failure_reason:
            print(f"Failure reason: {result.failure_reason}")
        if result.execution_result is not None:
            print(f"Execution status: {result.execution_result.status}")
            print(f"Execution summary: {result.execution_result.summary}")

        choice = input("\nRun environment setup graph again? [y/N]: ").strip().lower()
        if choice not in {"y", "yes"}:
            break


if __name__ == "__main__":
    main()

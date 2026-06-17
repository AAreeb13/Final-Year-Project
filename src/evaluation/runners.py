import json
import shutil
from pathlib import Path
from typing import Any, Callable

from src.agent_system_medium.graph import (
    AgentSystemMediumGraph,
    auto_approve,
    interactive_approve,
    print_agent_progress,
    quiet_agent_progress,
)
from src.agent_system_medium.schemas import MediumGraphResult
from src.evaluation.AgentSystem import AgentSystemRunner
from src.evaluation.project_store_converter import (
    PROJECT_STORE_ROOT,
    convert_project_store_to_system_run_output,
)
from src.evaluation.sdlc_eval_schema import (
    ComponentSpec,
    DirectorySpec,
    FileSpec,
    GraphDependencyEdgeSpec,
    GraphDependencySpec,
    HLComponentSpec,
    HLArchitectureSpec,
    ImplementationStep,
    ModuleSpec,
    RepositoryStructure,
    RequirementsSpec,
    TestPlan,
)
from src.evaluation.system_eval_schema import (
    ExecutionResult,
    GeneratedFile,
    SystemRunMetrics,
    SystemRunOutput,
    ToolCallRecord,
)
from src.multi_agent_system.store import ProjectStoreRepository
from src.single_agent_system.main import execute as single_agent_execute
from src.settings import settings


class SingleAgentSystemRunner(AgentSystemRunner):
    def __init__(self,  description: str):
        self.system_id = "single_agent_system"
        self.description = description
        self.system_prompt = "master"
        print("====== Initializing SingleAgentSystemRunner ======")
        print(f"====== Description: {description} \n====== System prompt: {self.system_prompt}")
        print("==================================================")

    def run(self, prompt: str, run_config: dict[str, Any]) -> tuple[dict[str, Any], str]:
        response, _response_json = single_agent_execute(
            problem_statement=prompt,
            prompt_name=self.system_prompt,
            allow_tool_execution=run_config.get("allow_tool_execution", False),
            debug_structured_output=run_config.get("debug_structured_output", False),
        )
        return response, f"run_{self.system_id}"

    def display_architecture(self) -> None:
        print(f"SingleAgentSystem using {self.system_prompt}") 


MediumGraphFactory = Callable[..., AgentSystemMediumGraph]
MultiAgentPipelineRunner = Callable[[str, dict[str, Any], ProjectStoreRepository], dict[str, Any]]


class MediumAgentSystemRunner(AgentSystemRunner):
    def __init__(
        self,
        description: str,
        *,
        graph_factory: MediumGraphFactory | None = None,
        graph_topology: str = "supervisor_centered",
        system_id: str = "medium_agent_system",
        run_id: str = "run_medium_agent",
        archive_generated_repo: bool = True,
        clear_repo_after_archive: bool = False,
    ):
        self.system_id = system_id
        self.run_id = run_id
        self.description = description
        self.graph_factory = graph_factory or AgentSystemMediumGraph
        self.graph_topology = graph_topology
        self.archive_generated_repo = archive_generated_repo
        self.clear_repo_after_archive = clear_repo_after_archive
        print("====== Initializing MediumAgentSystemRunner ======")
        print(f"====== Description: {description} \n====== Topology: {graph_topology}")
        print("==================================================")

    def run(self, prompt: str, run_config: dict[str, Any]) -> tuple[dict[str, Any], str]:
        human_approval = run_config.get("human_approval", False)
        progress = print_agent_progress if run_config.get("verbose", False) else quiet_agent_progress
        graph = self.graph_factory(
            approval_callback=interactive_approve if human_approval else auto_approve,
            progress_callback=progress,
            graph_topology=run_config.get("graph_topology", self.graph_topology),
        )
        raw_result = graph.run(
            user_prompt=prompt,
            max_repair_attempts=run_config.get("max_repair_attempts", 5),
            max_revision_attempts=run_config.get("max_revision_attempts", 3),
        )
        output = medium_graph_result_to_system_run_output(
            raw_result,
            project_id=run_config.get("project_id") or run_config.get("datapoint_id"),
            project_prompt=prompt,
            system_name=self.system_id,
            description=self.description,
        )
        return output.model_dump(), self.run_id

    def display_architecture(self) -> None:
        print(
            "MediumAgentSystem: supervisor plans requirements/design/file tasks, "
            "writer creates files, executor runs setup/tests."
        )

    def after_run_saved(
        self,
        *,
        artifact_dir: str | Path,
        output_path: str | Path,
        datapoint_id: str | None,
        run_id: str,
        run_config: dict[str, Any],
    ) -> None:
        archive_enabled = run_config.get("archive_generated_repo", self.archive_generated_repo)
        if not archive_enabled:
            return

        _move_generated_repo_after_save(
            artifact_dir=artifact_dir,
            output_path=output_path,
        )

    def reset_system(self, *, reason: str = "", verbose: bool = False) -> None:
        if verbose:
            detail = f" ({reason})" if reason else ""
            print(f"MediumAgentSystem reset{detail}: fresh graph will be built on next run")


class MultiAgentSystemRunner(AgentSystemRunner):
    def __init__(
        self,
        description: str,
        *,
        pipeline_runner: MultiAgentPipelineRunner | None = None,
        system_id: str = "multi_agent_system",
        run_id: str = "run_multi_agent",
        workspace_root: str | Path | None = None,
        archive_generated_repo: bool = True,
    ) -> None:
        self.system_id = system_id
        self.run_id = run_id
        self.description = description
        self.pipeline_runner = pipeline_runner or run_full_multi_agent_pipeline
        self.workspace_root = Path(workspace_root).expanduser().resolve() if workspace_root else None
        self.archive_generated_repo = archive_generated_repo
        self._last_workspace_root: Path | None = None
        print("====== Initializing MultiAgentSystemRunner ======")
        print(f"====== Description: {description}")
        print("=================================================")

    def run(self, prompt: str, run_config: dict[str, Any]) -> tuple[dict[str, Any], str]:
        project_id = _safe_project_id(
            str(run_config.get("project_id") or run_config.get("datapoint_id") or "evaluation_project")
        )
        workspace_value = run_config.get("workspace_root") or self.workspace_root or settings.WORKPLACE_FOLDER
        if workspace_value is None:
            raise ValueError("WORKPLACE_FOLDER or run_config['workspace_root'] is required for MultiAgentSystemRunner.")
        workspace_root = Path(
            workspace_value
        ).expanduser().resolve()
        self._last_workspace_root = workspace_root
        store_repository = ProjectStoreRepository(workspace_root=workspace_root)

        pipeline_error = ""
        try:
            self.pipeline_runner(
                prompt,
                {
                    **run_config,
                    "project_id": project_id,
                    "workspace_root": str(workspace_root),
                },
                store_repository,
            )
        except Exception as error:
            pipeline_error = f"Multi-agent pipeline failed with {type(error).__name__}: {error}"
            print(pipeline_error)

        project_dir = _project_store_dir(workspace_root, project_id)
        output = convert_project_store_to_system_run_output(
            project_dir,
            repo_root=_repo_root_from_workspace(workspace_root),
            system_name=self.system_id,
        )
        if pipeline_error:
            output.status = "failed"
            output.limitations = [*output.limitations, pipeline_error]
            output.summary = "\n".join(item for item in (output.summary, pipeline_error) if item)
        return output.model_dump(), self.run_id

    def after_run_saved(
        self,
        *,
        artifact_dir: str | Path,
        output_path: str | Path,
        datapoint_id: str | None,
        run_id: str,
        run_config: dict[str, Any],
    ) -> None:
        archive_enabled = run_config.get("archive_generated_repo", self.archive_generated_repo)
        if not archive_enabled:
            return
        _move_generated_repo_after_save(
            artifact_dir=artifact_dir,
            output_path=output_path,
            repo_root=_repo_root_from_workspace(self._last_workspace_root),
        )

    def reset_system(self, *, reason: str = "", verbose: bool = False) -> None:
        self._last_workspace_root = None
        result = _stop_project_container_for_reset()
        if verbose:
            detail = f" ({reason})" if reason else ""
            print(f"MultiAgentSystem reset{detail}: {result}")

    def display_architecture(self) -> None:
        print(
            "MultiAgentSystem: requirements -> design -> environment2 -> planning3 -> implementation2, "
            "normalised through project_store_converter."
        )


def run_full_multi_agent_pipeline(
    prompt: str,
    run_config: dict[str, Any],
    store_repository: ProjectStoreRepository,
) -> dict[str, Any]:
    from multi_agent_system.graphs.playground.environment2 import build_environment2_graph
    from multi_agent_system.graphs.playground.implementation2 import (
        _load_implementation2_inputs_from_store,
        build_implementation2_graph,
    )
    from src.multi_agent_system.graphs.planning import (
        _interactive_format_error_decision,
        _load_planning_inputs_from_store,
    )
    from multi_agent_system.graphs.playground.planning3 import build_planning3_graph
    from src.multi_agent_system.main import _require_stage_artifact
    from src.multi_agent_system.output_schema import (
        DesignTask,
        ImplementationExecutionResult,
        ImplementationStageOutput,
        PlanningStageOutput,
    )
    from src.multi_agent_system.graphs.design import build_design_graph
    from src.multi_agent_system.graphs.requirements import build_requirements_graph

    project_id = _safe_project_id(str(run_config["project_id"]))
    verbose = bool(run_config.get("verbose", False))
    _announce_multi_agent_stage(verbose, f"starting pipeline for {project_id}")
    if bool(run_config.get("overwrite_project_store", True)):
        project_dir = _project_store_dir(store_repository.workspace_root, project_id)
        if project_dir.exists():
            shutil.rmtree(project_dir)
    store = store_repository.create_project(
        project_id,
        prompt,
        overwrite=bool(run_config.get("overwrite_project_store", True)),
    )
    failures: list[str] = []

    _announce_multi_agent_stage(verbose, "requirements")
    try:
        requirements_result = _run_requirements_for_evaluation(
            prompt,
            max_rounds=int(run_config.get("max_requirements_rounds", 3)),
            graph_kwargs={
                "max_iterations": int(run_config.get("max_requirements_iterations", 2)),
                "model_name": run_config.get("requirements_model_name", "gpt-5.4-mini"),
            },
        )
        if requirements_result.stage_output is not None:
            store = store_repository.save_requirements_snapshot(
                project_id,
                requirements_result.stage_output,
                status="complete" if requirements_result.status == "complete" else "failed",
            )
        if requirements_result.stage_output is None or requirements_result.status != "complete":
            _record_pipeline_stage_failure(
                store_repository,
                project_id,
                "requirements",
                requirements_result.failure_reason or f"Requirements ended with status {requirements_result.status}",
                failures,
                verbose,
            )
    except Exception as error:
        _record_pipeline_stage_failure(
            store_repository,
            project_id,
            "requirements",
            _format_pipeline_error(error),
            failures,
            verbose,
        )
        store = store_repository.load_project(project_id)

    design_graph_kwargs = {
        "auto_approval": True,
        "approval_overrides_critic": True,
        "max_format_retries": int(run_config.get("max_design_format_retries", 3)),
        "model_name": run_config.get("design_model_name", "gpt-5.4-mini"),
        "verbose": verbose,
    }
    component_extraction = store.component_extraction
    if store.requirements is None:
        _record_pipeline_stage_skipped(
            store_repository,
            project_id,
            "design.extract_components",
            "Requirements output is missing.",
            failures,
            verbose,
        )
    else:
        _announce_multi_agent_stage(verbose, "design: extract_components")
        try:
            component_result = _run_design_task_for_evaluation(
                project_prompt=store.project_prompt,
                requirements=store.requirements,
                task=DesignTask(kind="extract_components"),
                graph_builder=lambda: build_design_graph(**design_graph_kwargs),
                max_revisions=int(run_config.get("max_design_revisions", 3)),
                verbose=verbose,
            )
            component_extraction = _require_stage_artifact(component_result).component_extraction
            if component_extraction is None:
                raise RuntimeError("Component extraction completed without component_extraction output")
            store = store_repository.save_component_extraction(project_id, component_extraction)
        except Exception as error:
            _record_pipeline_stage_failure(
                store_repository,
                project_id,
                "design.extract_components",
                _format_pipeline_error(error),
                failures,
                verbose,
            )
            store = store_repository.load_project(project_id)
            component_extraction = store.component_extraction

    previous_design_output = None
    for component in (component_extraction.components if component_extraction is not None else []):
        _announce_multi_agent_stage(verbose, f"design: decompose_component {component.name}")
        component_decomposition = None
        try:
            decomposition_result = _run_design_task_for_evaluation(
                project_prompt=store.project_prompt,
                requirements=store.requirements,
                task=DesignTask(kind="decompose_component", target_component=component.name),
                previous_design_output=previous_design_output,
                graph_builder=lambda: build_design_graph(**design_graph_kwargs),
                max_revisions=int(run_config.get("max_design_revisions", 3)),
                verbose=verbose,
            )
            component_decomposition = _require_stage_artifact(decomposition_result).component_decomposition
            if component_decomposition is None:
                raise RuntimeError(f"{component.name} decomposition completed without output")
            store = store_repository.save_component_decomposition(project_id, component.name, component_decomposition)
            previous_design_output = decomposition_result.stage_output
        except Exception as error:
            _record_pipeline_stage_failure(
                store_repository,
                project_id,
                f"design.decompose_component.{component.name}",
                _format_pipeline_error(error),
                failures,
                verbose,
            )
            store = store_repository.load_project(project_id)
            component_decomposition = store.component_decompositions.get(component.name)

        _announce_multi_agent_stage(verbose, f"design: design_modules {component.name}")
        if component_decomposition is None:
            _record_pipeline_stage_skipped(
                store_repository,
                project_id,
                f"design.design_modules.{component.name}",
                f"{component.name} decomposition output is missing.",
                failures,
                verbose,
            )
            continue
        try:
            module_result = _run_design_task_for_evaluation(
                project_prompt=store.project_prompt,
                requirements=store.requirements,
                task=DesignTask(kind="design_modules", target_component=component.name),
                previous_design_output=previous_design_output,
                graph_builder=lambda: build_design_graph(**design_graph_kwargs),
                max_revisions=int(run_config.get("max_design_revisions", 3)),
                verbose=verbose,
            )
            module_design = _require_stage_artifact(module_result).module_design
            if module_design is None:
                raise RuntimeError(f"{component.name} module design completed without output")
            store = store_repository.save_module_design(project_id, component.name, module_design)
            previous_design_output = module_result.stage_output
        except Exception as error:
            _record_pipeline_stage_failure(
                store_repository,
                project_id,
                f"design.design_modules.{component.name}",
                _format_pipeline_error(error),
                failures,
                verbose,
            )
            store = store_repository.load_project(project_id)

    _announce_multi_agent_stage(verbose, "environment2")
    try:
        environment_result = build_environment2_graph(
            store_repository=store_repository,
            verbose=verbose,
            format_error_callback=(
                _interactive_format_error_decision
                if bool(run_config.get("human_approval", False))
                else None
            ),
            max_format_retries=int(run_config.get("max_environment_format_retries", 3)),
        ).run(project_id=project_id)
        if environment_result.status != "complete":
            _record_pipeline_stage_failure(
                store_repository,
                project_id,
                "environment2",
                environment_result.failure_reason or f"Environment2 ended with status {environment_result.status}",
                failures,
                verbose,
            )
    except Exception as error:
        _record_pipeline_stage_failure(
            store_repository,
            project_id,
            "environment2",
            _format_pipeline_error(error),
            failures,
            verbose,
        )

    _announce_multi_agent_stage(verbose, "planning3")
    try:
        planning_inputs = _load_planning_inputs_from_store(project_id, store_repository)
        planning_graph_inputs = {key: value for key, value in planning_inputs.items() if key != "project_id"}
        planning_result = build_planning3_graph(
            format_error_callback=(
                _interactive_format_error_decision
                if bool(run_config.get("human_approval", False))
                else None
            ),
            max_format_retries=int(run_config.get("max_planning_format_retries", 3)),
            min_concrete_test_case_ratio=float(run_config.get("min_concrete_test_case_ratio", 0.75)),
            verbose=verbose,
            debug=bool(run_config.get("debug_planning", False)),
        ).run(
            **planning_graph_inputs,
            max_iterations=int(run_config.get("max_planning_iterations", 3)),
        )
        if planning_result.stage_output is not None:
            store_repository.save_planning_snapshot(
                project_id,
                planning_result.stage_output,
                status="complete" if planning_result.stage_output.approved else "failed",
            )
        if planning_result.stage_output is None or not planning_result.stage_output.approved:
            _record_pipeline_stage_failure(
                store_repository,
                project_id,
                "planning3",
                planning_result.failure_reason or f"Planning3 ended with status {planning_result.status}",
                failures,
                verbose,
            )
    except Exception as error:
        _record_pipeline_stage_failure(
            store_repository,
            project_id,
            "planning3",
            _format_pipeline_error(error),
            failures,
            verbose,
        )

    _announce_multi_agent_stage(verbose, "implementation2")
    try:
        implementation_inputs = _load_implementation2_inputs_from_store(project_id, store_repository)
        implementation_result = build_implementation2_graph(
            store_repository=store_repository,
            verbose=verbose,
            max_repair_attempts=int(run_config.get("max_implementation_repairs", 3)),
        ).run(**implementation_inputs)
        if implementation_result.stage_output is not None:
            store_repository.save_implementation(project_id, implementation_result.stage_output)
        if implementation_result.stage_output is None or not implementation_result.stage_output.approved:
            _record_pipeline_stage_failure(
                store_repository,
                project_id,
                "implementation2",
                implementation_result.failure_reason or f"Implementation2 ended with status {implementation_result.status}",
                failures,
                verbose,
            )
    except Exception as error:
        failure_reason = _format_pipeline_error(error)
        store_repository.save_implementation(
            project_id,
            ImplementationStageOutput(
                approved=False,
                failure_reason=failure_reason,
                execution=ImplementationExecutionResult(
                    status="failed",
                    summary=failure_reason,
                ),
            ),
        )
        _record_pipeline_stage_failure(
            store_repository,
            project_id,
            "implementation2",
            failure_reason,
            failures,
            verbose,
        )

    return {
        "project_id": project_id,
        "status": "failed" if failures else "complete",
        "failures": failures,
    }


def _announce_multi_agent_stage(verbose: bool, stage: str) -> None:
    if verbose:
        print(f"Multi-agent pipeline stage: {stage}")


def _record_pipeline_stage_failure(
    store_repository: ProjectStoreRepository,
    project_id: str,
    stage: str,
    reason: str,
    failures: list[str],
    verbose: bool,
) -> None:
    message = f"{stage}: {reason}"
    failures.append(message)
    store_repository.update_stage_status(project_id, stage, "failed")
    store_repository.update_stage_status(project_id, f"{stage}.failure_reason", reason)
    if verbose:
        print(f"Multi-agent pipeline stage failed: {message}")


def _record_pipeline_stage_skipped(
    store_repository: ProjectStoreRepository,
    project_id: str,
    stage: str,
    reason: str,
    failures: list[str],
    verbose: bool,
) -> None:
    message = f"{stage}: skipped because {reason}"
    failures.append(message)
    store_repository.update_stage_status(project_id, stage, "skipped")
    store_repository.update_stage_status(project_id, f"{stage}.failure_reason", reason)
    if verbose:
        print(f"Multi-agent pipeline stage skipped: {message}")


def _format_pipeline_error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _stop_project_container_for_reset() -> str:
    try:
        from src.tools.docker_code_execution.tool import stop_container

        raw_result = stop_container.invoke({})
        if isinstance(raw_result, str):
            try:
                parsed = json.loads(raw_result)
            except json.JSONDecodeError:
                return raw_result
            message = parsed.get("message") if isinstance(parsed, dict) else None
            status = parsed.get("status") if isinstance(parsed, dict) else None
            if message and status:
                return f"{status}: {message}"
            if message:
                return str(message)
            return raw_result
        return str(raw_result)
    except Exception as error:
        return f"container reset skipped: {type(error).__name__}: {error}"


def _run_requirements_for_evaluation(
    project_prompt: str,
    *,
    max_rounds: int,
    graph_kwargs: dict[str, Any],
):
    from src.multi_agent_system.graphs.requirements import build_requirements_graph
    from src.multi_agent_system.output_schema import QuestionAnswer

    graph = build_requirements_graph(**graph_kwargs)
    current = graph.run(project_prompt)
    rounds = 1
    while current.status == "waiting_for_user" and rounds < max_rounds:
        context = list(current.question_answer_context)
        for question in current.questions:
            context.append(
                QuestionAnswer(
                    question=question,
                    answer=(
                        "No additional user input is available during evaluation; "
                        "make a conservative assumption from the project prompt."
                    ),
                )
            )
        current = graph.run(
            project_prompt,
            previous_requirements=current.requirements,
            question_answer_context=context,
        )
        rounds += 1
    return current


def _run_design_task_for_evaluation(
    *,
    project_prompt: str,
    requirements: Any,
    task: Any,
    graph_builder: Callable[[], Any],
    previous_design_output: Any = None,
    max_revisions: int,
    verbose: bool = False,
):
    graph = graph_builder()
    if verbose:
        print(f"Design task start: {_format_design_task_for_log(task)}")
    current = graph.run(
        project_prompt=project_prompt,
        requirements=requirements,
        task=task,
        previous_design_output=previous_design_output,
    )
    if verbose:
        print(f"Design task result: status={current.status}")
    revision_count = 0
    while current.status != "complete":
        if current.status == "failed":
            raise RuntimeError(current.failure_reason or current.critic_output)
        revision_count += 1
        if revision_count > max_revisions:
            if current.stage_output is None:
                raise RuntimeError(f"Cannot force-complete {task.kind} without a design stage output")
            current.stage_output.approved = True
            current.status = "complete"
            if verbose:
                print(
                    "Design task forced complete: "
                    f"{_format_design_task_for_log(task)} revisions={revision_count - 1}/{max_revisions}"
                )
            return current
        if verbose:
            print(
                "Design revision: "
                f"{_format_design_task_for_log(task)} "
                f"round={revision_count}/{max_revisions}"
            )
        current = graph.run(
            project_prompt=project_prompt,
            requirements=requirements,
            task=task,
            previous_design_output=current.stage_output or previous_design_output,
            question_answer_context=current.question_answer_context,
            critic_feedback=_design_revision_feedback(current),
        )
        if verbose:
            print(f"Design task result: status={current.status}")
    return current


def _design_revision_feedback(current: Any) -> str:
    critic_output = getattr(current, "critic_output", None)
    if critic_output is None:
        return ""
    feedback_parts: list[str] = []
    if getattr(critic_output, "feedback", ""):
        feedback_parts.append(critic_output.feedback)
    if getattr(critic_output, "required_changes", []):
        feedback_parts.append("Required changes:\n- " + "\n- ".join(critic_output.required_changes))
    if getattr(critic_output, "questions", []):
        feedback_parts.append(
            "Do not ask the user questions during evaluation. Resolve these with conservative design decisions:\n- "
            + "\n- ".join(critic_output.questions)
        )
    return "\n\n".join(feedback_parts)


def _format_design_task_for_log(task: Any) -> str:
    kind = getattr(task, "kind", str(task))
    target = getattr(task, "target_component", None)
    if target:
        return f"{kind} target={target}"
    return str(kind)


def _project_store_dir(workspace_root: Path, project_id: str) -> Path:
    return workspace_root / PROJECT_STORE_ROOT / project_id


def _repo_root_from_workspace(workspace_root: Path | None) -> Path | None:
    if workspace_root is None or settings.REPO_NAME is None:
        return None
    return (workspace_root.expanduser().resolve() / settings.REPO_NAME).resolve()


def _safe_project_id(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value)
    return cleaned.strip("._-") or "evaluation_project"


def medium_graph_result_to_system_run_output(
    result: dict[str, Any] | MediumGraphResult,
    *,
    project_id: str | None,
    project_prompt: str,
    system_name: str = "multi_agent_system",
    description: str = "",
) -> SystemRunOutput:
    medium_result = (
        result
        if isinstance(result, MediumGraphResult)
        else MediumGraphResult.model_validate(result)
    )
    supervisor = medium_result.supervisor_output
    executor_output = medium_result.executor_output

    requirements = RequirementsSpec(
        functional=list(supervisor.functional_requirements),
        non_functional=list(supervisor.non_functional_requirements),
        constraints=list(supervisor.constraints),
        assumptions=list(supervisor.assumptions),
        out_of_scope=list(supervisor.out_of_scope),
    )
    components = _components_from_medium_result(medium_result)
    modules = _modules_from_medium_result(medium_result)
    module_dependency_graph = _dependency_graph_from_medium_result(medium_result, modules)
    repository_structure = _repository_structure_from_medium_result(medium_result)
    implementation_plan = _implementation_steps_from_medium_result(medium_result)
    test_plan = _test_plan_from_medium_result(medium_result)
    generated_files = _generated_files_from_medium_result(medium_result)
    execution_results = _execution_results_from_medium_result(medium_result)
    tool_calls = _tool_calls_from_medium_result(medium_result)
    failed_execution = [execution for execution in execution_results if not execution.success]
    failed_tools = [tool_call for tool_call in tool_calls if not tool_call.success]
    status = "complete" if medium_result.status == "success" else "failed"
    limitations = []
    if any(not generated_file.content for generated_file in generated_files):
        limitations.append(
            "Some medium agent writer outputs did not expose full generated file contents; generated_files.content is left empty for those files."
        )
    if not supervisor.architecture_design:
        limitations.append("Medium agent supervisor did not provide explicit architecture design details.")
    if not executor_output:
        limitations.append("Medium agent executor output was missing; execution_results may be empty.")

    return SystemRunOutput(
        project_id=project_id,
        system_name=system_name,
        status=status,
        requirements=requirements,
        high_level_design=HLArchitectureSpec(
            style="; ".join(supervisor.architecture_design[:3]) if supervisor.architecture_design else None,
            components=[
                HLComponentSpec(
                    name=component.name,
                    responsibilities=list(component.responsibilities),
                )
                for component in components
            ],
        ),
        components=components,
        modules=modules,
        module_dependency_graph=module_dependency_graph,
        repository_structure=repository_structure,
        implementation_plan=implementation_plan,
        test_plan=test_plan,
        generated_files=generated_files,
        execution_results=execution_results,
        tool_calls=tool_calls,
        metrics=SystemRunMetrics(
            total_steps=len(medium_result.trace),
            agent_messages=len(medium_result.trace),
            tool_calls=len(tool_calls),
            successful_tool_calls=len(tool_calls) - len(failed_tools),
            failed_tool_calls=len(failed_tools),
            execution_commands=len(execution_results),
            failed_execution_commands=len(failed_execution),
            repair_attempts=medium_result.attempts,
        ),
        summary="\n".join(
            item
            for item in (
                description,
                supervisor.project_summary,
                executor_output.summary if executor_output else "",
            )
            if item
        ),
        limitations=limitations,
    )


def _components_from_medium_result(result: MediumGraphResult) -> list[ComponentSpec]:
    supervisor = result.supervisor_output
    components = [
        ComponentSpec(name=item, responsibilities=[item])
        for item in supervisor.architecture_design
        if item
    ]
    if components:
        return components
    names = sorted(
        {
            _component_name_from_path(task.relative_path)
            for task in supervisor.file_tasks
            if task.relative_path
        }
    )
    return [ComponentSpec(name=name) for name in names if name]


def _modules_from_medium_result(result: MediumGraphResult) -> list[ModuleSpec]:
    modules: list[ModuleSpec] = []
    task_name_by_id = {
        task.task_id: Path(task.relative_path).stem or task.task_id
        for task in result.supervisor_output.file_tasks
    }
    for task in result.supervisor_output.file_tasks:
        modules.append(
            ModuleSpec(
                name=Path(task.relative_path).stem or task.task_id,
                component=_component_name_from_path(task.relative_path),
                responsibilities=[task.description] if task.description else [],
                dependencies=[
                    task_name_by_id.get(dependency, dependency)
                    for dependency in task.depends_on
                ],
            )
        )
    return modules


def _dependency_graph_from_medium_result(
    result: MediumGraphResult,
    modules: list[ModuleSpec],
) -> GraphDependencySpec:
    task_name_by_id = {
        task.task_id: Path(task.relative_path).stem or task.task_id
        for task in result.supervisor_output.file_tasks
    }
    edges: list[GraphDependencyEdgeSpec] = []
    for task in result.supervisor_output.file_tasks:
        target = task_name_by_id.get(task.task_id, task.task_id)
        for dependency in task.depends_on:
            edges.append(
                GraphDependencyEdgeSpec(
                    source=task_name_by_id.get(dependency, dependency),
                    target=target,
                )
            )
    return GraphDependencySpec(
        nodes=[module.name for module in modules if module.name],
        edges=edges,
    )


def _repository_structure_from_medium_result(result: MediumGraphResult) -> RepositoryStructure:
    directories: dict[str, DirectorySpec] = {}
    for task in result.supervisor_output.file_tasks:
        if not task.relative_path:
            continue
        path = Path(task.relative_path)
        parent = str(path.parent) if str(path.parent) != "." else "."
        directory = directories.setdefault(
            parent,
            DirectorySpec(
                name=Path(parent).name if parent != "." else ".",
                parent=str(Path(parent).parent) if parent not in {".", ""} else None,
            ),
        )
        directory.files.append(FileSpec(name=path.name, modules=[path.stem]))
    return RepositoryStructure(directories=list(directories.values()))


def _generated_files_from_medium_result(result: MediumGraphResult) -> list[GeneratedFile]:
    generated_files: list[GeneratedFile] = []
    for writer_output in result.writer_outputs:
        generated_files.append(
            GeneratedFile(
                path=writer_output.relative_path,
                content=_extract_generated_content(writer_output.raw_agent_state),
                purpose=writer_output.summary or None,
            )
        )
    return generated_files


def _implementation_steps_from_medium_result(result: MediumGraphResult) -> list[ImplementationStep]:
    writer_summaries = {
        writer_output.task_id: writer_output.summary
        for writer_output in result.writer_outputs
    }
    return [
        ImplementationStep(
            step_id=task.task_id,
            module_target=Path(task.relative_path).stem,
            action=f"{task.action} {task.relative_path}",
            result=writer_summaries.get(task.task_id) or task.description,
        )
        for task in result.supervisor_output.file_tasks
    ]


def _test_plan_from_medium_result(result: MediumGraphResult) -> TestPlan:
    supervisor_plan = result.supervisor_output.test_plan
    commands = [" ".join(command) for command in supervisor_plan.test_commands]
    return TestPlan(
        unit_tests=[supervisor_plan.notes] if supervisor_plan.notes else [],
        integration_tests=[],
        commands=commands,
        testing_framework=supervisor_plan.testing_framework,
    )


def _execution_results_from_medium_result(result: MediumGraphResult) -> list[ExecutionResult]:
    executor_output = result.executor_output
    if executor_output is None:
        return []
    return [
        ExecutionResult(
            command=" ".join(command_result.command),
            exit_code=command_result.exit_code,
            stdout=command_result.stdout,
            stderr=command_result.stderr,
            success=command_result.status == "success",
            summary=command_result.message,
        )
        for command_result in executor_output.command_results
    ]


def _tool_calls_from_medium_result(result: MediumGraphResult) -> list[ToolCallRecord]:
    tool_calls: list[ToolCallRecord] = []
    for event in result.trace:
        approved = event.approval.approved if event.approval else True
        output_status = str(event.output.get("status") or "")
        output_failed = output_status in {"error", "failed"}
        success = approved and not output_failed
        tool_calls.append(
            ToolCallRecord(
                tool_name=event.agent,
                args_summary=event.step,
                success=success,
                result_summary=_summarise_medium_output(event.output),
                error=None
                if success
                else (event.approval.message if event.approval and not approved else _summarise_medium_output(event.output)),
            )
        )
    for execution in _execution_results_from_medium_result(result):
        tool_calls.append(
            ToolCallRecord(
                tool_name="executor_command",
                args_summary=execution.command,
                success=execution.success,
                result_summary=execution.summary or execution.stdout,
                error=execution.stderr if not execution.success else None,
            )
        )
    return tool_calls


def _component_name_from_path(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) >= 2:
        return parts[-2]
    return "application"


def _extract_generated_content(raw_agent_state: dict[str, Any]) -> str:
    for key in ("content", "file_content", "generated_content", "code"):
        value = raw_agent_state.get(key)
        if isinstance(value, str):
            return value

    generated_files = raw_agent_state.get("generated_files")
    if isinstance(generated_files, list):
        for file_item in generated_files:
            if isinstance(file_item, dict):
                value = file_item.get("content")
                if isinstance(value, str):
                    return value

    messages = raw_agent_state.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            content = getattr(message, "content", None)
            if isinstance(message, dict):
                content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content

    return ""


def _move_generated_repo_after_save(
    *,
    artifact_dir: str | Path,
    output_path: str | Path,
    repo_root: Path | None = None,
) -> None:
    artifact_dir = Path(artifact_dir)
    output_path = Path(output_path)
    limitations: list[str] = []
    repo_root = repo_root or _resolve_generated_repo_root(limitations)
    if repo_root is None:
        _append_output_limitations(output_path, limitations)
        return

    archive_path = artifact_dir / "generated_repo"
    moved = _move_generated_repo_snapshot(repo_root, archive_path, limitations)
    if moved:
        print(f"Moved generated repo files to: {archive_path}")

    if limitations:
        _append_output_limitations(output_path, limitations)


def _resolve_generated_repo_root(limitations: list[str]) -> Path | None:
    if settings.WORKPLACE_FOLDER is None:
        limitations.append("Could not archive generated repo because WORKPLACE_FOLDER is not configured.")
        return None
    if settings.REPO_NAME is None:
        limitations.append("Could not archive generated repo because REPO_NAME is not configured.")
        return None

    workspace_root = Path(settings.WORKPLACE_FOLDER).expanduser().resolve()
    repo_root = (workspace_root / settings.REPO_NAME).expanduser().resolve()
    if repo_root != workspace_root and workspace_root not in repo_root.parents:
        limitations.append("Could not archive generated repo because the repo path is outside WORKPLACE_FOLDER.")
        return None
    if not repo_root.exists() or not repo_root.is_dir():
        limitations.append(f"Could not archive generated repo because the repo path does not exist: {repo_root}")
        return None
    return repo_root


def _move_generated_repo_snapshot(
    repo_root: Path,
    archive_path: Path,
    limitations: list[str],
) -> bool:
    try:
        if archive_path.exists():
            shutil.rmtree(archive_path)
        archive_path.mkdir(parents=True)
        for child in list(repo_root.iterdir()):
            if child.name == ".git":
                continue
            shutil.move(str(child), str(archive_path / child.name))
        return True
    except Exception as error:
        limitations.append(f"Could not move generated repo files to {archive_path}: {error}")
        return False


def _append_output_limitations(output_path: Path, limitations: list[str]) -> None:
    if not limitations:
        return

    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        existing = data.get("limitations")
        if not isinstance(existing, list):
            existing = []
        data["limitations"] = [*existing, *limitations]
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as error:
        print(f"Warning: could not record archive limitation in {output_path}: {error}")
        for limitation in limitations:
            print(f"Warning: {limitation}")


def _summarise_medium_output(output: dict[str, Any]) -> str:
    for key in ("summary", "project_summary", "status", "message"):
        value = output.get(key)
        if value:
            return str(value)
    return ", ".join(output.keys())

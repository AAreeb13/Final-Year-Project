from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.sdlc_eval_schema import (
    ComponentSpec,
    DirectorySpec,
    FileSpec,
    GraphDependencyEdgeSpec,
    GraphDependencySpec,
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
from src.settings import settings


PROJECT_STORE_ROOT = ".multi_agent_system/projects"


class ProjectStoreConversion:
    def __init__(
        self,
        *,
        project_dir: Path,
        output: SystemRunOutput,
        output_path: Path,
        limitations: list[str],
    ) -> None:
        self.project_dir = project_dir
        self.output = output
        self.output_path = output_path
        self.limitations = limitations


def convert_project_store_to_system_run_output(
    project_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    system_name: str = "multi_agent_system",
) -> SystemRunOutput:
    project_path = Path(project_dir).expanduser().resolve()
    limitations: list[str] = []

    project_store = _load_json(project_path / "project_store.json", limitations)
    planning = _load_json(project_path / "planning.json", limitations)
    implementation = _load_json(project_path / "implementation.json", limitations)

    project_id = str(project_store.get("project_id") or project_path.name)
    repository_root = _resolve_repo_root(repo_root, limitations)

    requirements = _convert_requirements(project_store, limitations)
    high_level_design = _convert_high_level_design(project_store, limitations)
    components = _convert_components(project_store, limitations)
    modules = _convert_modules(project_store, limitations)
    dependency_graph = _convert_dependency_graph(project_store, modules, limitations)
    repository_structure = _convert_repository_structure(project_store, planning, implementation, limitations)
    implementation_plan = _convert_implementation_plan(planning, limitations)
    test_plan = _convert_test_plan(planning, limitations)
    generated_files = _convert_generated_files(implementation, repository_root, limitations)
    execution_results = _convert_execution_results(implementation, limitations)
    tool_calls = _convert_tool_calls(project_store, implementation)
    metrics = _build_metrics(project_store, implementation, generated_files, execution_results, tool_calls)

    status = _status_from_store(project_store, implementation, planning)
    summary = _summary_from_store(project_store, planning, implementation)
    limitations = _dedupe(limitations)

    return SystemRunOutput(
        project_id=project_id,
        system_name=system_name,
        status=status,
        requirements=requirements,
        high_level_design=high_level_design,
        components=components,
        modules=modules,
        module_dependency_graph=dependency_graph,
        repository_structure=repository_structure,
        implementation_plan=implementation_plan,
        test_plan=test_plan,
        generated_files=generated_files,
        execution_results=execution_results,
        tool_calls=tool_calls,
        metrics=metrics,
        summary=summary,
        limitations=limitations,
    )


def convert_project_store_to_output_json(
    project_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    evaluation_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    system_name: str = "multi_agent_system",
    run_id: str | None = None,
) -> ProjectStoreConversion:
    project_path = Path(project_dir).expanduser().resolve()
    output = convert_project_store_to_system_run_output(
        project_path,
        repo_root=repo_root,
        system_name=system_name,
    )
    save_path = _resolve_output_path(
        output_path=output_path,
        evaluation_dir=evaluation_dir,
        project_id=output.project_id or project_path.name,
        system_name=system_name,
        run_id=run_id,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    return ProjectStoreConversion(
        project_dir=project_path,
        output=output,
        output_path=save_path,
        limitations=output.limitations,
    )


def _load_json(path: Path, limitations: list[str]) -> dict[str, Any]:
    if not path.exists():
        limitations.append(f"Missing {path.name}; related fields were left empty.")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        limitations.append(f"Could not read {path.name}: {error}; related fields were left empty.")
        return {}


def _convert_requirements(project_store: dict[str, Any], limitations: list[str]) -> RequirementsSpec:
    raw_requirements = _get(project_store, "requirements", "requirements") or {}
    if not raw_requirements:
        limitations.append("Requirements were missing from project_store.json.")
    return RequirementsSpec(
        functional=_as_str_list(raw_requirements.get("functional_requirements")),
        non_functional=_as_str_list(raw_requirements.get("non_functional_requirements")),
        constraints=_as_str_list(raw_requirements.get("constraints")),
        assumptions=_as_str_list(raw_requirements.get("assumptions")),
        out_of_scope=_as_str_list(raw_requirements.get("out_of_scope")),
    )


def _convert_high_level_design(project_store: dict[str, Any], limitations: list[str]) -> HLArchitectureSpec:
    raw_design = _get(project_store, "component_extraction", "high_level_architecture") or {}
    if not raw_design:
        limitations.append("High-level architecture was missing; left high_level_design empty.")
    return HLArchitectureSpec.model_validate(raw_design or {})


def _convert_components(project_store: dict[str, Any], limitations: list[str]) -> list[ComponentSpec]:
    raw_components = _get(project_store, "component_extraction", "components") or []
    if not raw_components:
        decompositions = project_store.get("component_decompositions") or {}
        raw_components = [
            decomposition.get("component")
            for decomposition in decompositions.values()
            if isinstance(decomposition, dict) and decomposition.get("component")
        ]
        if raw_components:
            limitations.append("Components were reconstructed from component_decompositions.")
    if not raw_components:
        limitations.append("Components were missing from project_store.json.")
    return [ComponentSpec.model_validate(component) for component in raw_components if isinstance(component, dict)]


def _convert_modules(project_store: dict[str, Any], limitations: list[str]) -> list[ModuleSpec]:
    module_designs = project_store.get("module_designs") or {}
    modules: list[ModuleSpec] = []
    for design in module_designs.values():
        if not isinstance(design, dict):
            continue
        for module in design.get("modules") or []:
            if isinstance(module, dict):
                modules.append(ModuleSpec.model_validate(module))
    if not modules:
        decompositions = project_store.get("component_decompositions") or {}
        for decomposition in decompositions.values():
            if not isinstance(decomposition, dict):
                continue
            for module in decomposition.get("modules") or []:
                if isinstance(module, dict):
                    modules.append(ModuleSpec.model_validate(module))
        if modules:
            limitations.append("Modules were reconstructed from component_decompositions because module_designs were unavailable.")
    if not modules:
        limitations.append("Modules were missing from project_store.json.")
    return modules


def _convert_dependency_graph(
    project_store: dict[str, Any],
    modules: list[ModuleSpec],
    limitations: list[str],
) -> GraphDependencySpec:
    nodes: list[str] = []
    edges: list[GraphDependencyEdgeSpec] = []
    module_designs = project_store.get("module_designs") or {}
    for design in module_designs.values():
        if not isinstance(design, dict):
            continue
        graph = design.get("dependency_graph") or {}
        nodes.extend(_as_str_list(graph.get("nodes")))
        for edge in graph.get("edges") or []:
            if isinstance(edge, dict):
                edges.append(GraphDependencyEdgeSpec.model_validate(edge))

    if not nodes:
        nodes = [module.name for module in modules if module.name]
    if not edges:
        for module in modules:
            for dependency in module.dependencies:
                edges.append(GraphDependencyEdgeSpec(source=dependency, target=module.name))
        if edges:
            limitations.append("Dependency graph edges were reconstructed from module dependencies.")
    return GraphDependencySpec(nodes=_dedupe(nodes), edges=_dedupe_edges(edges))


def _convert_repository_structure(
    project_store: dict[str, Any],
    planning: dict[str, Any],
    implementation: dict[str, Any],
    limitations: list[str],
) -> RepositoryStructure:
    raw_structure = project_store.get("repository_structure")
    if isinstance(raw_structure, dict) and raw_structure.get("directories"):
        return RepositoryStructure.model_validate(raw_structure)

    paths = [
        file_plan.get("relative_path")
        for file_plan in _get(planning, "implementation_plan", "files") or []
        if isinstance(file_plan, dict)
    ]
    paths.extend(
        write.get("relative_path")
        for write in _get(implementation, "execution", "file_writes") or []
        if isinstance(write, dict)
    )
    structure = _repository_structure_from_paths([path for path in paths if path])
    if structure.directories:
        limitations.append("Repository structure was reconstructed from planned/written file paths.")
    else:
        limitations.append("Repository structure was missing and could not be reconstructed.")
    return structure


def _convert_implementation_plan(planning: dict[str, Any], limitations: list[str]) -> list[ImplementationStep]:
    steps: list[ImplementationStep] = []
    for step in _get(planning, "implementation_plan", "steps") or []:
        if not isinstance(step, dict):
            continue
        steps.append(
            ImplementationStep(
                step_id=str(step.get("step_id") or ""),
                module_target=", ".join(_as_str_list(step.get("target_modules"))),
                action=str(step.get("action") or ""),
                result=str(step.get("description") or step.get("target_file_id") or ""),
            )
        )

    if steps:
        return steps

    for file_plan in _get(planning, "implementation_plan", "files") or []:
        if not isinstance(file_plan, dict):
            continue
        steps.append(
            ImplementationStep(
                step_id=str(file_plan.get("file_id") or ""),
                module_target=", ".join(_as_str_list(file_plan.get("modules"))),
                action=f"Create {file_plan.get('relative_path') or 'implementation file'}",
                result=str(file_plan.get("purpose") or ""),
            )
        )
    if steps:
        limitations.append("Implementation steps were reconstructed from implementation file plans.")
    else:
        limitations.append("Implementation plan was missing from planning.json.")
    return steps


def _convert_test_plan(planning: dict[str, Any], limitations: list[str]) -> TestPlan:
    graph = planning.get("test_plan") or {}
    unit_tests: list[str] = []
    integration_tests: list[str] = []
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        rendered = _render_test_node(node)
        if not rendered:
            continue
        if node.get("kind") == "unit":
            unit_tests.append(rendered)
        else:
            integration_tests.append(rendered)
    if not unit_tests and not integration_tests:
        limitations.append("Test plan nodes were missing from planning.json.")
    return TestPlan(
        unit_tests=unit_tests,
        integration_tests=integration_tests,
        commands=_as_str_list(graph.get("commands")),
        testing_framework=str(graph.get("testing_framework") or ""),
    )


def _convert_generated_files(
    implementation: dict[str, Any],
    repo_root: Path | None,
    limitations: list[str],
) -> list[GeneratedFile]:
    files: list[GeneratedFile] = []
    seen: set[str] = set()
    for write in _get(implementation, "execution", "file_writes") or []:
        if not isinstance(write, dict):
            continue
        relative_path = str(write.get("relative_path") or "")
        if not relative_path or relative_path in seen:
            continue
        seen.add(relative_path)
        content = ""
        if repo_root is not None:
            content = _read_generated_file(repo_root, relative_path, limitations)
        files.append(
            GeneratedFile(
                path=relative_path,
                content=content,
                purpose=str(write.get("summary") or write.get("message") or "") or None,
            )
        )
    if not files:
        limitations.append("Generated files were missing from implementation.json.")
    elif repo_root is None:
        limitations.append("Generated file contents were left empty because repo_root could not be resolved.")
    return files


def _convert_execution_results(implementation: dict[str, Any], limitations: list[str]) -> list[ExecutionResult]:
    results: list[ExecutionResult] = []
    for execution in _get(implementation, "execution", "test_executions") or []:
        if not isinstance(execution, dict):
            continue
        status = str(execution.get("status") or "")
        results.append(
            ExecutionResult(
                command=" ".join(_as_str_list(execution.get("command"))),
                exit_code=execution.get("exit_code"),
                stdout=str(execution.get("stdout") or ""),
                stderr=str(execution.get("stderr") or ""),
                success=status == "success" or execution.get("exit_code") == 0,
                summary=str(execution.get("summary") or execution.get("message") or ""),
            )
        )
    if not results:
        limitations.append("Execution/test results were missing from implementation.json.")
    return results


def _convert_tool_calls(project_store: dict[str, Any], implementation: dict[str, Any]) -> list[ToolCallRecord]:
    calls: list[ToolCallRecord] = []
    for result in _get(project_store, "environment_setup_execution", "results") or []:
        if not isinstance(result, dict):
            continue
        calls.append(
            ToolCallRecord(
                tool_name=str(result.get("tool_name") or result.get("step") or "environment_setup"),
                args_summary=" ".join(_as_str_list(result.get("command"))) or str(result.get("path") or ""),
                success=str(result.get("status") or "") == "success",
                result_summary=str(result.get("summary") or result.get("message") or ""),
                error=str(result.get("stderr") or "") or None,
            )
        )
    for write in _get(implementation, "execution", "file_writes") or []:
        if not isinstance(write, dict):
            continue
        calls.append(
            ToolCallRecord(
                tool_name="file_write",
                args_summary=str(write.get("relative_path") or ""),
                success=str(write.get("status") or "") == "success",
                result_summary=str(write.get("summary") or write.get("message") or ""),
                error=None if str(write.get("status") or "") == "success" else str(write.get("message") or ""),
            )
        )
    return calls


def _build_metrics(
    project_store: dict[str, Any],
    implementation: dict[str, Any],
    generated_files: list[GeneratedFile],
    execution_results: list[ExecutionResult],
    tool_calls: list[ToolCallRecord],
) -> SystemRunMetrics:
    failed_commands = [result for result in execution_results if not result.success]
    failed_tool_calls = [call for call in tool_calls if not call.success]
    repair_attempts = max(0, len(_get(implementation, "execution", "file_writes") or []) - len(generated_files))
    return SystemRunMetrics(
        total_steps=len(project_store.get("stage_statuses") or {}),
        tool_calls=len(tool_calls),
        successful_tool_calls=len(tool_calls) - len(failed_tool_calls),
        failed_tool_calls=len(failed_tool_calls),
        execution_commands=len(execution_results),
        failed_execution_commands=len(failed_commands),
        repair_attempts=repair_attempts,
    )


def _status_from_store(project_store: dict[str, Any], implementation: dict[str, Any], planning: dict[str, Any]) -> str:
    stage_statuses = project_store.get("stage_statuses") or {}
    if any(status == "failed" for status in stage_statuses.values()):
        return "failed"
    implementation_status = _get(implementation, "execution", "status")
    if implementation_status:
        return str(implementation_status)
    if implementation.get("approved") is True:
        return "complete"
    if planning.get("approved") is True:
        return "planned"
    return str((project_store.get("stage_statuses") or {}).get("implementation") or "unknown")


def _summary_from_store(project_store: dict[str, Any], planning: dict[str, Any], implementation: dict[str, Any]) -> str:
    parts = []
    if project_store.get("project_prompt"):
        parts.append(f"Project prompt: {project_store['project_prompt']}")
    stage_statuses = project_store.get("stage_statuses") or {}
    failed_reasons = [
        f"{key.removesuffix('.failure_reason')}: {value}"
        for key, value in stage_statuses.items()
        if key.endswith(".failure_reason") and value
    ]
    if failed_reasons:
        parts.append("Stage failures:\n- " + "\n- ".join(failed_reasons))
    if planning.get("critic_verdict"):
        parts.append(f"Planning verdict: {planning['critic_verdict']}")
    if _get(implementation, "execution", "summary"):
        parts.append(f"Implementation summary: {_get(implementation, 'execution', 'summary')}")
    if implementation.get("failure_reason"):
        parts.append(f"Implementation failure: {implementation['failure_reason']}")
    return "\n".join(parts)


def _repository_structure_from_paths(paths: list[str]) -> RepositoryStructure:
    directories: dict[str, DirectorySpec] = {}
    for raw_path in paths:
        path = Path(raw_path)
        parent = str(path.parent) if str(path.parent) != "." else "."
        directory = directories.setdefault(parent, DirectorySpec(name=Path(parent).name if parent != "." else ".", parent=str(Path(parent).parent) if parent not in {".", ""} else None))
        directory.files.append(FileSpec(name=path.name))
    return RepositoryStructure(directories=list(directories.values()))


def _render_test_node(node: dict[str, Any]) -> str:
    title = str(node.get("title") or node.get("node_id") or "")
    modules = ", ".join(_as_str_list(node.get("target_modules")))
    signatures = ", ".join(_as_str_list(node.get("target_signatures")))
    cases = []
    for case in node.get("test_cases") or []:
        if isinstance(case, str):
            cases.append(case)
        elif isinstance(case, dict):
            case_text = case.get("assertion_summary") or case.get("expected_output_summary") or case.get("name")
            if case_text:
                cases.append(str(case_text))
    details = []
    if modules:
        details.append(f"modules: {modules}")
    if signatures:
        details.append(f"signatures: {signatures}")
    if cases:
        details.append("cases: " + "; ".join(cases[:3]))
    return f"{title} ({'; '.join(details)})" if details else title


def _read_generated_file(repo_root: Path, relative_path: str, limitations: list[str]) -> str:
    try:
        file_path = (repo_root / relative_path).resolve()
        if file_path != repo_root and repo_root not in file_path.parents:
            limitations.append(f"Skipped generated file content outside repo root: {relative_path}")
            return ""
        if not file_path.exists() or not file_path.is_file():
            limitations.append(f"Generated file content missing on disk: {relative_path}")
            return ""
        return file_path.read_text(encoding="utf-8")
    except Exception as error:
        limitations.append(f"Could not read generated file {relative_path}: {error}")
        return ""


def _resolve_repo_root(repo_root: str | Path | None, limitations: list[str]) -> Path | None:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    if settings.WORKPLACE_FOLDER and settings.REPO_NAME:
        return (Path(settings.WORKPLACE_FOLDER).expanduser() / settings.REPO_NAME).resolve()
    limitations.append("WORKPLACE_FOLDER/REPO_NAME were not configured; generated file contents may be empty.")
    return None


def _resolve_output_path(
    *,
    output_path: str | Path | None,
    evaluation_dir: str | Path | None,
    project_id: str,
    system_name: str,
    run_id: str | None,
) -> Path:
    if output_path is not None:
        return Path(output_path).expanduser().resolve()
    root = Path(evaluation_dir).expanduser().resolve() if evaluation_dir else Path.cwd() / "converted_evaluation_runs"
    if evaluation_dir is None and output_path is None:
        if settings.EVALUATION_DIRECTORY is None:
            root = Path.cwd() / "converted_evaluation_runs"
        else:
            root = Path(settings.EVALUATION_DIRECTORY).expanduser().resolve()

    eval_dir = _next_eval_dir(root)
    run_name = run_id or "run_multi_agent"
    return eval_dir / project_id / system_name / run_name / "output.json"


def _next_eval_dir(evaluation_root: Path) -> Path:
    existing_numbers: list[int] = []
    if evaluation_root.exists():
        for path in evaluation_root.iterdir():
            if not path.is_dir() or not path.name.startswith("eval_"):
                continue
            try:
                existing_numbers.append(int(path.name.replace("eval_", "")))
            except ValueError:
                continue
    next_number = max(existing_numbers, default=0) + 1
    return evaluation_root / f"eval_{next_number:03d}"


def _project_dir_from_id(project_id: str, workspace_root: str | Path | None) -> Path:
    root_value = workspace_root or settings.WORKPLACE_FOLDER
    if root_value is None:
        raise ValueError("workspace_root or WORKPLACE_FOLDER is required when using --project-id.")
    return Path(root_value).expanduser().resolve() / PROJECT_STORE_ROOT / project_id


def _get(data: dict[str, Any], *path: str) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and not isinstance(item, dict)]
    return [str(value)]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _dedupe_edges(edges: list[GraphDependencyEdgeSpec]) -> list[GraphDependencyEdgeSpec]:
    seen: set[tuple[str, str]] = set()
    deduped: list[GraphDependencyEdgeSpec] = []
    for edge in edges:
        key = (edge.source, edge.target)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert multi-agent project store artifacts into EvaluationAnalyser output.json."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--project-dir", help="Directory containing project_store.json/planning.json/implementation.json.")
    source_group.add_argument("--project-id", help="Project id under WORKPLACE_FOLDER/.multi_agent_system/projects.")
    parser.add_argument("--workspace-root", help="Workspace root used with --project-id.")
    parser.add_argument("--repo-root", help="Repository root used to read generated file contents.")
    parser.add_argument("--output-json", help="Exact output.json path to write.")
    parser.add_argument("--evaluation-dir", help="Evaluation root where an analyser-compatible run directory is created.")
    parser.add_argument("--system-name", default="multi_agent_system")
    parser.add_argument("--run-id", help="Optional run directory name when --evaluation-dir is used.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_dir = (
        Path(args.project_dir).expanduser().resolve()
        if args.project_dir
        else _project_dir_from_id(args.project_id, args.workspace_root)
    )
    conversion = convert_project_store_to_output_json(
        project_dir,
        output_path=args.output_json,
        evaluation_dir=args.evaluation_dir,
        repo_root=args.repo_root,
        system_name=args.system_name,
        run_id=args.run_id,
    )
    print(f"Wrote EvaluationAnalyser output to: {conversion.output_path}")
    if conversion.limitations:
        print("Conversion limitations:")
        for limitation in conversion.limitations:
            print(f"- {limitation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

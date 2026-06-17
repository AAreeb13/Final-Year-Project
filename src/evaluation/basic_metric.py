from __future__ import annotations

import re

from src.evaluation.sdlc_eval_schema import (
    GraphDependencyEdgeSpec,
    ProjectSpec,
    RepositoryStructure,
)
from src.evaluation.system_eval_schema import ExecutionResult, SystemRunOutput


def compute_basic_metrics(project: ProjectSpec, result: SystemRunOutput) -> dict[str, int | float | str]:
    """Compute deterministic thesis-aligned metrics from reference and run output."""

    expected_requirements = [
        *project.requirements.functional,
        *project.requirements.non_functional,
    ]
    actual_requirements = [
        *result.requirements.functional,
        *result.requirements.non_functional,
    ]
    expected_components = [component.name for component in project.components]
    actual_components = [component.name for component in result.components]
    expected_modules = [module.name for module in project.modules]
    actual_modules = [module.name for module in result.modules]
    expected_dependencies = _normalised_edges(project.module_dependency_graph.edges)
    actual_dependencies = _normalised_edges(result.module_dependency_graph.edges)
    expected_repo_paths = _repository_paths(project.repository_structure)
    actual_repo_paths = _repository_paths(result.repository_structure)
    if not actual_repo_paths:
        actual_repo_paths = _generated_file_paths(result)

    requirement_overlap = _overlap_count(expected_requirements, actual_requirements)
    constraint_overlap = _overlap_count(project.requirements.constraints, result.requirements.constraints)
    component_overlap = _overlap_count(expected_components, actual_components)
    module_overlap = _overlap_count(expected_modules, actual_modules)
    dependency_overlap = len(expected_dependencies & actual_dependencies)
    repository_overlap = len(expected_repo_paths & actual_repo_paths)

    test_results = [execution for execution in result.execution_results if _is_test_command(execution)]
    build_results = [execution for execution in result.execution_results if not _is_test_command(execution)]
    failed_execution_count = len([execution for execution in result.execution_results if not execution.success])
    failed_tool_calls = len([tool_call for tool_call in result.tool_calls if not tool_call.success])
    test_count = len(result.test_plan.unit_tests) + len(result.test_plan.integration_tests)
    completion_rate = 100 if _is_complete_status(result.status) else 0
    iteration_count = result.metrics.total_steps or result.metrics.agent_messages

    return {
        "project_id": project.project_id,
        "system_name": result.system_name,
        "implementation_success": result.status,
        "completion_rate": completion_rate,
        "iteration_count": iteration_count,
        "total_requirements": len(expected_requirements),
        "implemented_requirements": len(actual_requirements),
        "matched_requirements": requirement_overlap,
        "requirement_recall": _percentage(requirement_overlap, len(expected_requirements)),
        "requirement_precision": _percentage(requirement_overlap, len(actual_requirements)),
        "total_constraints": len(project.requirements.constraints),
        "captured_constraints": constraint_overlap,
        "constraint_capture_rate": _percentage(constraint_overlap, len(project.requirements.constraints)),
        "total_components": len(project.components),
        "implemented_components": len(result.components),
        "matched_components": component_overlap,
        "component_recall": _percentage(component_overlap, len(expected_components)),
        "component_precision": _percentage(component_overlap, len(actual_components)),
        "total_modules": len(project.modules),
        "implemented_modules": len(result.modules),
        "matched_modules": module_overlap,
        "module_recall": _percentage(module_overlap, len(expected_modules)),
        "module_precision": _percentage(module_overlap, len(actual_modules)),
        "total_dependencies": len(expected_dependencies),
        "matched_dependencies": dependency_overlap,
        "dependency_recall": _percentage(dependency_overlap, len(expected_dependencies)),
        "total_repository_paths": len(expected_repo_paths),
        "matched_repository_paths": repository_overlap,
        "repository_structure_accuracy": _percentage(repository_overlap, len(expected_repo_paths)),
        "build_success_rate": _success_rate(build_results),
        "test_pass_rate": _success_rate(test_results),
        "test_coverage_score": _percentage(test_count, len(expected_requirements)),
        "total_tool_calls": len(result.tool_calls),
        "successful_tool_calls": len(result.tool_calls) - failed_tool_calls,
        "failed_tool_calls": failed_tool_calls,
        "tool_call_success_rate": _success_rate(result.tool_calls),
        "execution_commands": len(result.execution_results),
        "execution_failure_count": failed_execution_count,
        "repair_attempts": result.metrics.repair_attempts,
    }


def _is_complete_status(status: str) -> bool:
    return _normalise_text(status) in {"complete", "completed", "success", "succeeded"}


def _overlap_count(expected: list[str], actual: list[str]) -> int:
    return len(_normalised_set(expected) & _normalised_set(actual))


def _normalised_set(values: list[str]) -> set[str]:
    return {_normalise_text(value) for value in values if _normalise_text(value)}


def _normalise_text(value: str) -> str:
    value = str(value).casefold().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _normalised_edges(edges: list[GraphDependencyEdgeSpec]) -> set[tuple[str, str]]:
    return {
        (_normalise_text(edge.source), _normalise_text(edge.target))
        for edge in edges
        if _normalise_text(edge.source) and _normalise_text(edge.target)
    }


def _repository_paths(repository: RepositoryStructure) -> set[str]:
    paths: set[str] = set()
    for directory in repository.directories:
        directory_path = _directory_path(directory.name, directory.parent)
        if directory_path and directory_path != ".":
            paths.add(directory_path)
        for file_item in directory.files:
            if not file_item.name:
                continue
            if directory_path in {"", "."}:
                paths.add(_normalise_path(file_item.name))
            else:
                paths.add(_normalise_path(f"{directory_path}/{file_item.name}"))
    return {path for path in paths if path}


def _generated_file_paths(result: SystemRunOutput) -> set[str]:
    paths: set[str] = set()
    for file in result.generated_files:
        if not file.path:
            continue
        normalised_path = _normalise_path(file.path)
        parts = normalised_path.split("/")
        for index in range(1, len(parts)):
            paths.add("/".join(parts[:index]))
        paths.add(normalised_path)
    return {path for path in paths if path}


def _directory_path(name: str, parent: str | None) -> str:
    if not parent:
        return _normalise_path(name or ".")
    if parent == ".":
        return _normalise_path(name)
    return _normalise_path(f"{parent}/{name}")


def _normalise_path(path: str) -> str:
    return str(path).replace("\\", "/").strip().strip("/")


def _is_test_command(execution: ExecutionResult) -> bool:
    command = execution.command.casefold()
    return any(
        marker in command
        for marker in (
            "pytest",
            "unittest",
            "npm test",
            "yarn test",
            "pnpm test",
            "jest",
            "vitest",
            "mocha",
            "cargo test",
            "go test",
        )
    )


def _success_rate(items: list) -> float:
    if not items:
        return 0
    successes = len([item for item in items if getattr(item, "success", False)])
    return _percentage(successes, len(items))


def _percentage(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0
    return round((min(numerator, denominator) / denominator) * 100, 2)

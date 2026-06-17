from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import shorten
from typing import Any, Callable, Sequence

from src.evaluation.basic_metric import compute_basic_metrics
from src.evaluation.DataLoader import DataLoader
from src.evaluation.sdlc_eval_schema import (
    GraphDependencySpec,
    HLArchitectureSpec,
    ModuleSpec,
    ProjectSpec,
    RepositoryStructure,
    RequirementsSpec,
    TestPlan,
)
from src.evaluation.system_eval_schema import SystemRunOutput


InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], None]


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    category: str
    title: str
    max_score: float
    unit: str
    guidance: str


MANUAL_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        metric_id="requirement_extraction.requirements_recall",
        category="Requirement Extraction",
        title="Requirements Recall",
        max_score=100,
        unit="percent",
        guidance="How much of the reference functional and non-functional requirements were recovered?",
    ),
    MetricDefinition(
        metric_id="requirement_extraction.requirements_precision",
        category="Requirement Extraction",
        title="Requirements Precision",
        max_score=100,
        unit="percent",
        guidance="How much of the generated requirements are valid against the reference task?",
    ),
    MetricDefinition(
        metric_id="requirement_extraction.constraint_capture_rate",
        category="Requirement Extraction",
        title="Constraint Capture Rate",
        max_score=100,
        unit="percent",
        guidance="How many reference constraints were captured by the system?",
    ),
    MetricDefinition(
        metric_id="architecture.component_recall",
        category="Quantitative Architecture",
        title="Component Recall",
        max_score=100,
        unit="percent",
        guidance="How many reference components were recovered?",
    ),
    MetricDefinition(
        metric_id="architecture.component_precision",
        category="Quantitative Architecture",
        title="Component Precision",
        max_score=100,
        unit="percent",
        guidance="How many generated components are valid reference components?",
    ),
    MetricDefinition(
        metric_id="architecture.module_recall",
        category="Quantitative Architecture",
        title="Module Recall",
        max_score=100,
        unit="percent",
        guidance="How many reference modules were recovered?",
    ),
    MetricDefinition(
        metric_id="architecture.module_precision",
        category="Quantitative Architecture",
        title="Module Precision",
        max_score=100,
        unit="percent",
        guidance="How many generated modules are valid reference modules?",
    ),
    MetricDefinition(
        metric_id="architecture.dependency_recall",
        category="Quantitative Architecture",
        title="Dependency Recall",
        max_score=100,
        unit="percent",
        guidance="How many expected dependency edges were recovered?",
    ),
    MetricDefinition(
        metric_id="architecture.requirement_to_component_coverage",
        category="Quantitative Architecture",
        title="Requirement-to-Component Coverage",
        max_score=100,
        unit="percent",
        guidance="How many reference requirements are traceably represented by generated components?",
    ),
    MetricDefinition(
        metric_id="repository.repository_structure_accuracy",
        category="Repository Structure",
        title="Repository Structure Accuracy",
        max_score=100,
        unit="percent",
        guidance="How closely does the generated repository structure match the reference structure?",
    ),
    MetricDefinition(
        metric_id="repository.repository_standard",
        category="Repository Structure",
        title="Repository Standard",
        max_score=10,
        unit="score",
        guidance="How well organised, conventional, and navigable is the generated repository?",
    ),
    MetricDefinition(
        metric_id="execution.build_success_rate",
        category="Execution",
        title="Build Success Rate",
        max_score=100,
        unit="percent",
        guidance="What proportion of build/setup commands succeeded without manual fixes?",
    ),
    MetricDefinition(
        metric_id="execution.test_pass_rate",
        category="Execution",
        title="Test Pass Rate",
        max_score=100,
        unit="percent",
        guidance="What proportion of test commands passed?",
    ),
    MetricDefinition(
        metric_id="test_coverage.test_coverage_score",
        category="Test Coverage",
        title="Test Coverage Score",
        max_score=100,
        unit="percent",
        guidance="How much of the expected behaviour is covered by the generated or planned tests?",
    ),
    MetricDefinition(
        metric_id="system_efficiency.completion_rate",
        category="System Efficiency",
        title="Completion Rate",
        max_score=100,
        unit="percent",
        guidance="Did the system produce a complete final project output for the task?",
    ),
    MetricDefinition(
        metric_id="system_efficiency.iteration_count",
        category="System Efficiency",
        title="Iteration Count",
        max_score=1000,
        unit="count",
        guidance="How many agent iterations or major steps were required?",
    ),
    MetricDefinition(
        metric_id="system_efficiency.tool_call_count",
        category="System Efficiency",
        title="Tool Call Count",
        max_score=1000,
        unit="count",
        guidance="How many external tool calls were made during generation, implementation, testing, or repair?",
    ),
    MetricDefinition(
        metric_id="system_efficiency.execution_failure_count",
        category="System Efficiency",
        title="Execution Failure Count",
        max_score=1000,
        unit="count",
        guidance="How many build, run, or test attempts failed?",
    ),
    MetricDefinition(
        metric_id="system_efficiency.repair_success_rate",
        category="System Efficiency",
        title="Repair Success Rate",
        max_score=100,
        unit="percent",
        guidance="What proportion of detected errors were successfully repaired by the system?",
    ),
    MetricDefinition(
        metric_id="system_efficiency.trace_consistency",
        category="System Efficiency",
        title="Trace Consistency",
        max_score=100,
        unit="percent",
        guidance="How consistent is the final implementation with the earlier requirements, design, and implementation plan?",
    ),
    MetricDefinition(
        metric_id="software_design.solid_violations",
        category="Quantitative Architecture",
        title="SOLID Principle Violations",
        max_score=1000,
        unit="count",
        guidance="How many clear SOLID principle violations are present?",
    ),
    MetricDefinition(
        metric_id="high_level_architecture.architecture_elements",
        category="High Level Architecture",
        title="Components, Relationships, and Technology",
        max_score=10,
        unit="score",
        guidance="Did it identify the expected components, relationships, and relevant technologies?",
    ),
    MetricDefinition(
        metric_id="high_level_architecture.modules_in_components",
        category="High Level Architecture",
        title="Modules in Components",
        max_score=10,
        unit="score",
        guidance="Did it identify the important modules and place them in sensible components?",
    ),
    MetricDefinition(
        metric_id="high_level_architecture.module_dependencies",
        category="High Level Architecture",
        title="Dependencies Between Modules",
        max_score=10,
        unit="score",
        guidance="Did it identify meaningful dependencies between the modules?",
    ),
    MetricDefinition(
        metric_id="software_design.design_principles",
        category="Software Design",
        title="Design Principles",
        max_score=10,
        unit="score",
        guidance="Did the design avoid clear violations of general design principles?",
    ),
    MetricDefinition(
        metric_id="software_design.solid",
        category="Software Design",
        title="SOLID",
        max_score=10,
        unit="score",
        guidance="Did the design respect SOLID principles where they apply?",
    ),
    MetricDefinition(
        metric_id="software_design.design_patterns",
        category="Software Design",
        title="Design Patterns Used and Identified",
        max_score=10,
        unit="score",
        guidance="Were useful design patterns used or identified without being forced?",
    ),
    MetricDefinition(
        metric_id="software_design.separation_of_concerns",
        category="Software Design",
        title="Separation of Concerns at Module Level",
        max_score=10,
        unit="score",
        guidance="Are module responsibilities distinct and appropriately scoped?",
    ),
    MetricDefinition(
        metric_id="software_design.coupling",
        category="Software Design",
        title="Coupling",
        max_score=10,
        unit="score",
        guidance="Are dependencies limited, justified, and free from unnecessary cycles?",
    ),
    MetricDefinition(
        metric_id="software_design.cohesion",
        category="Software Design",
        title="Cohesion",
        max_score=10,
        unit="score",
        guidance="Does each component or module have a focused and internally consistent responsibility?",
    ),
    MetricDefinition(
        metric_id="software_design.maintainability",
        category="Software Design",
        title="Maintainability",
        max_score=10,
        unit="score",
        guidance="Is the design and implementation understandable and likely to support future change?",
    ),
    MetricDefinition(
        metric_id="code_hygiene.code_hygiene",
        category="Code Hygiene",
        title="Code Hygiene",
        max_score=10,
        unit="score",
        guidance="Judge naming, structure, clarity, and repository organisation. Generated code is not displayed by this CLI.",
    ),
    MetricDefinition(
        metric_id="test_coverage.test_coverage",
        category="Test Coverage",
        title="Test Coverage",
        max_score=10,
        unit="score",
        guidance="Did the output include a useful testing strategy or tests for expected behaviours?",
    ),
)


class EvaluationAnalyser:
    def __init__(
        self,
        data_loader: DataLoader,
        evaluation_dir: str | Path | None = None,
        *,
        output: OutputFunc = print,
    ):
        self.data_loader = data_loader
        self.evaluation_dir = (
            Path(evaluation_dir).expanduser().resolve()
            if evaluation_dir is not None
            else None
        )
        self.output = output

    def discover_runs(self, evaluation_dir: str | Path | None = None) -> list[Path]:
        root = (
            Path(evaluation_dir).expanduser().resolve()
            if evaluation_dir is not None
            else self.evaluation_dir
        )

        if root is None:
            raise ValueError("An evaluation directory is required when --output-json is not supplied.")

        if not root.exists() or not root.is_dir():
            raise ValueError(f"Evaluation directory does not exist: {root}")

        return sorted(root.glob("eval_*/*/*/*/output.json"))

    def choose_run(
        self,
        runs: Sequence[Path],
        *,
        input_func: InputFunc = input,
    ) -> Path:
        if not runs:
            raise ValueError("No output.json files were found.")

        if len(runs) == 1:
            selected = runs[0]
            self.output(f"Using only discovered run: {selected}")
            return selected

        self.output("Available evaluation runs:")
        for index, run_path in enumerate(runs, start=1):
            ids = self.extract_run_metadata(run_path)
            label = (
                f"{ids['eval_id']} / {ids['datapoint_id']} / "
                f"{ids['system_id']} / {ids['run_id']}"
            )
            self.output(f"  {index}. {label}")

        while True:
            raw_choice = input_func(f"Choose a run [1-{len(runs)}]: ").strip()
            try:
                choice = int(raw_choice)
            except ValueError:
                self.output("Please enter a number.")
                continue

            if 1 <= choice <= len(runs):
                return runs[choice - 1]

            self.output(f"Please enter a number between 1 and {len(runs)}.")

    def analyse(
        self,
        output_json: str | Path,
        *,
        input_func: InputFunc = input,
    ) -> Path:
        output_path = Path(output_json).expanduser().resolve()
        project, result = self.load_comparison_inputs(output_path)

        comparison = self.format_comparison(project, result)
        if comparison:
            self.output(comparison)

        automated_metrics = compute_basic_metrics(project, result)
        self.output(self.format_automated_metrics(automated_metrics))

        manual_metrics = self.collect_manual_metrics(input_func=input_func)
        evaluation = self.build_evaluation_summary(
            output_path=output_path,
            project=project,
            result=result,
            manual_metrics=manual_metrics,
        )
        return self.save_evaluation(output_path, evaluation)

    def load_comparison_inputs(self, output_json: str | Path) -> tuple[ProjectSpec, SystemRunOutput]:
        output_path = Path(output_json).expanduser().resolve()
        if not output_path.exists() or not output_path.is_file():
            raise ValueError(f"output.json does not exist: {output_path}")

        with output_path.open("r", encoding="utf-8") as f:
            raw_output = json.load(f)

        result = SystemRunOutput.model_validate(raw_output)
        metadata = self.extract_run_metadata(output_path)
        datapoint_id = result.project_id or metadata["datapoint_id"]

        if not datapoint_id:
            raise ValueError(
                "Could not determine datapoint id from output.json or its run path."
            )

        project = self.data_loader.retrieve_full_datapoint(datapoint_id)
        return project, result

    def collect_manual_metrics(
        self,
        *,
        input_func: InputFunc = input,
        metrics: Sequence[MetricDefinition] = MANUAL_METRICS,
    ) -> dict[str, dict[str, Any]]:
        manual_metrics: dict[str, dict[str, Any]] = {}

        self.output("")
        self.output("Manual evaluation")
        self.output("Enter a score and short qualitative note for each metric.")

        current_category = ""
        for metric in metrics:
            if metric.category != current_category:
                current_category = metric.category
                self.output("")
                self.output(current_category)

            self.output(f"{metric.title}: {metric.guidance}")
            score = self.prompt_score(metric, input_func=input_func)
            feedback = input_func(f"{metric.title} feedback: ").strip()

            manual_metrics[metric.metric_id] = {
                "category": metric.category,
                "title": metric.title,
                "score": score,
                "max_score": metric.max_score,
                "unit": metric.unit,
                "qualitative_feedback": feedback,
                "guidance": metric.guidance,
            }

        return manual_metrics

    def prompt_score(
        self,
        metric: MetricDefinition,
        *,
        input_func: InputFunc = input,
    ) -> int | float:
        while True:
            raw_score = input_func(
                f"{metric.title} score [0-{_format_number(metric.max_score)}"
                f"{'%' if metric.unit == 'percent' else ''}]: "
            ).strip()

            try:
                score = float(raw_score)
            except ValueError:
                self.output("Please enter a numeric score.")
                continue

            if 0 <= score <= metric.max_score:
                return int(score) if score.is_integer() else score

            self.output(f"Please enter a score between 0 and {_format_number(metric.max_score)}.")

    def build_evaluation_summary(
        self,
        *,
        output_path: Path,
        project: ProjectSpec,
        result: SystemRunOutput,
        manual_metrics: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        metadata = self.extract_run_metadata(output_path)
        automated_metrics = compute_basic_metrics(project, result)

        return {
            "schema_version": "1.0",
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "eval_id": metadata["eval_id"],
                "datapoint_id": metadata["datapoint_id"] or project.project_id,
                "project_id": project.project_id,
                "system_id": metadata["system_id"],
                "run_id": metadata["run_id"],
                "system_name": result.system_name,
                "status": result.status,
                "dataset_path": str(self.data_loader.data_path),
                "output_path": str(output_path),
            },
            "automated_metrics": automated_metrics,
            "manual_metrics": manual_metrics,
        }

    def format_automated_metrics(self, metrics: dict[str, Any]) -> str:
        metric_order = (
            "requirement_recall",
            "requirement_precision",
            "constraint_capture_rate",
            "component_recall",
            "component_precision",
            "module_recall",
            "module_precision",
            "dependency_recall",
            "repository_structure_accuracy",
            "build_success_rate",
            "test_pass_rate",
            "test_coverage_score",
            "completion_rate",
            "iteration_count",
            "tool_call_success_rate",
            "execution_failure_count",
            "repair_attempts",
        )
        lines = ["", "Automated Metrics", "-----------------"]
        for key in metric_order:
            if key not in metrics:
                continue
            lines.append(f"{_humanise_metric_key(key)}: {_format_metric_value(metrics[key], key)}")
        return "\n".join(lines)

    def save_evaluation(self, output_path: str | Path, evaluation: dict[str, Any]) -> Path:
        output_path = Path(output_path).expanduser().resolve()
        save_path = self.next_evaluation_path(output_path.parent)
        save_path.write_text(
            json.dumps(evaluation, indent=2),
            encoding="utf-8",
        )
        self.output(f"Saved evaluation summary to: {save_path}")
        return save_path

    @staticmethod
    def next_evaluation_path(run_dir: str | Path) -> Path:
        run_dir = Path(run_dir).expanduser().resolve()
        first_path = run_dir / "evaluation.json"
        if not first_path.exists():
            return first_path

        index = 2
        while True:
            candidate = run_dir / f"evaluation_{index:03d}.json"
            if not candidate.exists():
                return candidate
            index += 1

    @staticmethod
    def extract_run_metadata(output_path: str | Path) -> dict[str, str]:
        output_path = Path(output_path).expanduser().resolve()
        run_dir = output_path.parent

        metadata = {
            "eval_id": "",
            "datapoint_id": "",
            "system_id": "",
            "run_id": run_dir.name,
        }

        try:
            metadata["system_id"] = run_dir.parent.name
            metadata["datapoint_id"] = run_dir.parent.parent.name
            metadata["eval_id"] = run_dir.parent.parent.parent.name
        except IndexError:
            pass

        return metadata

    def format_comparison(self, project: ProjectSpec, result: SystemRunOutput) -> str:
        sections = [
            self._format_requirements(project.requirements, result.requirements),
            self._format_architecture(project.high_level_design, result.high_level_design),
            self._format_modules(project.modules, result.modules),
            self._format_dependency_graph(
                project.module_dependency_graph,
                result.module_dependency_graph,
            ),
            self._format_repository_structure(
                project.repository_structure,
                result.repository_structure,
            ),
            self._format_test_plan(project.test_plan, result.test_plan),
            self._format_generated_files(result),
        ]

        return "\n\n".join(section for section in sections if section)

    def _format_requirements(
        self,
        reference: RequirementsSpec,
        output: RequirementsSpec,
    ) -> str:
        return _render_side_by_side(
            "Requirements",
            _requirements_lines(reference),
            _requirements_lines(output),
        )

    def _format_architecture(
        self,
        reference: HLArchitectureSpec,
        output: HLArchitectureSpec,
    ) -> str:
        return _render_side_by_side(
            "High Level Architecture",
            _architecture_lines(reference),
            _architecture_lines(output),
        )

    def _format_modules(
        self,
        reference: Sequence[ModuleSpec],
        output: Sequence[ModuleSpec],
    ) -> str:
        return _render_side_by_side(
            "Modules",
            _module_lines(reference),
            _module_lines(output),
        )

    def _format_dependency_graph(
        self,
        reference: GraphDependencySpec,
        output: GraphDependencySpec,
    ) -> str:
        return _render_side_by_side(
            "Module Dependency Graph",
            _dependency_lines(reference),
            _dependency_lines(output),
        )

    def _format_repository_structure(
        self,
        reference: RepositoryStructure,
        output: RepositoryStructure,
    ) -> str:
        return _render_side_by_side(
            "Repository Structure",
            _repository_lines(reference),
            _repository_lines(output),
        )

    def _format_test_plan(self, reference: TestPlan, output: TestPlan) -> str:
        return _render_side_by_side(
            "Test Plan",
            _test_plan_lines(reference),
            _test_plan_lines(output),
        )

    def _format_generated_files(self, result: SystemRunOutput) -> str:
        paths = [file.path for file in result.generated_files if file.path]
        return _render_one_column("Generated Files", "Output", paths)


def _requirements_lines(requirements: RequirementsSpec) -> list[str]:
    lines: list[str] = []
    for title, values in (
        ("Functional", requirements.functional),
        ("Non-functional", requirements.non_functional),
        ("Constraints", requirements.constraints),
        ("Assumptions", requirements.assumptions),
        ("Out of scope", requirements.out_of_scope),
    ):
        if values:
            lines.append(f"{title}:")
            lines.extend(f"- {value}" for value in values)
    return lines


def _architecture_lines(architecture: HLArchitectureSpec) -> list[str]:
    lines: list[str] = []
    if architecture.style:
        lines.append(f"Style: {architecture.style}")

    if architecture.components:
        lines.append("Components:")
        for component in architecture.components:
            responsibility = "; ".join(component.responsibilities)
            detail = f" ({component.type})" if component.type else ""
            if responsibility:
                detail = f"{detail}: {responsibility}"
            lines.append(f"- {component.name}{detail}")

    if architecture.relationships:
        lines.append("Relationships:")
        for relationship in architecture.relationships:
            label = relationship.relationship_type or "relates to"
            lines.append(f"- {relationship.source} -> {relationship.target} ({label})")

    if architecture.technologies:
        lines.append("Technologies:")
        lines.extend(f"- {technology}" for technology in architecture.technologies)

    return lines


def _module_lines(modules: Sequence[ModuleSpec]) -> list[str]:
    lines: list[str] = []
    for module in modules:
        heading = module.name
        if module.component:
            heading = f"{heading} [{module.component}]"
        if module.type:
            heading = f"{heading} ({module.type})"
        lines.append(f"- {heading}")

        if module.responsibilities:
            lines.append(f"  responsibilities: {'; '.join(module.responsibilities)}")
        if module.dependencies:
            lines.append(f"  dependencies: {', '.join(module.dependencies)}")
        if module.signatures:
            signatures = ", ".join(signature.name for signature in module.signatures if signature.name)
            if signatures:
                lines.append(f"  signatures: {signatures}")

    return lines


def _dependency_lines(graph: GraphDependencySpec) -> list[str]:
    lines: list[str] = []
    if graph.nodes:
        lines.append("Nodes:")
        lines.extend(f"- {node}" for node in graph.nodes)
    if graph.edges:
        lines.append("Edges:")
        lines.extend(f"- {edge.source} -> {edge.target}" for edge in graph.edges)
    return lines


def _repository_lines(repository: RepositoryStructure) -> list[str]:
    lines: list[str] = []
    for directory in repository.directories:
        directory_name = _directory_display_name(directory.name, directory.parent)
        lines.append(f"- {directory_name}/")
        for file_item in directory.files:
            module_text = f" ({', '.join(file_item.modules)})" if file_item.modules else ""
            lines.append(f"  - {file_item.name}{module_text}")
    return lines


def _test_plan_lines(test_plan: TestPlan) -> list[str]:
    lines: list[str] = []
    if test_plan.testing_framework:
        lines.append(f"Framework: {test_plan.testing_framework}")
    if test_plan.unit_tests:
        lines.append("Unit tests:")
        lines.extend(f"- {test}" for test in test_plan.unit_tests)
    if test_plan.integration_tests:
        lines.append("Integration tests:")
        lines.extend(f"- {test}" for test in test_plan.integration_tests)
    if test_plan.commands:
        lines.append("Commands:")
        lines.extend(f"- {command}" for command in test_plan.commands)
    return lines


def _directory_display_name(name: str, parent: str | None) -> str:
    if not parent:
        return name or "."
    if parent == ".":
        return name
    return f"{parent}/{name}"


def _render_side_by_side(
    title: str,
    reference_lines: Sequence[str],
    output_lines: Sequence[str],
) -> str:
    reference_lines = [line for line in reference_lines if line]
    output_lines = [line for line in output_lines if line]

    if not reference_lines and not output_lines:
        return ""
    if not reference_lines:
        return _render_one_column(title, "Output", output_lines)
    if not output_lines:
        return _render_one_column(title, "Dataset", reference_lines)

    width = 58
    divider = " | "
    header = f"{'Dataset'.ljust(width)}{divider}Output"
    rule = f"{'-' * width}{divider}{'-' * width}"
    rows = [f"\n{title}", header, rule]

    max_rows = max(len(reference_lines), len(output_lines))
    for index in range(max_rows):
        left = _shorten_line(reference_lines[index], width) if index < len(reference_lines) else ""
        right = _shorten_line(output_lines[index], width) if index < len(output_lines) else ""
        rows.append(f"{left.ljust(width)}{divider}{right}")

    return "\n".join(rows)


def _render_one_column(title: str, label: str, lines: Sequence[str]) -> str:
    lines = [line for line in lines if line]
    if not lines:
        return ""

    rendered = [f"\n{title}", label, "-" * len(label)]
    rendered.extend(lines)
    return "\n".join(rendered)


def _shorten_line(value: str, width: int) -> str:
    return shorten(value, width=width, placeholder="...")


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _humanise_metric_key(key: str) -> str:
    return key.replace("_", " ").title()


def _format_metric_value(value: Any, key: str) -> str:
    if isinstance(value, float):
        rendered = _format_number(value)
    else:
        rendered = str(value)
    if key.endswith("_rate") or key.endswith("_recall") or key.endswith("_precision") or key.endswith("_accuracy") or key == "test_coverage_score":
        return f"{rendered}%"
    return rendered


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare an evaluation output.json with its dataset reference and collect manual feedback."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the Revolutionising-SWE-dataset folder.",
    )
    parser.add_argument(
        "--evaluation-dir",
        help="Path to the directory containing eval_*/.../output.json runs.",
    )
    parser.add_argument(
        "--output-json",
        help="Path to a specific output.json file to analyse.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.output_json and not args.evaluation_dir:
        raise ValueError("Provide either --output-json or --evaluation-dir.")

    data_loader = DataLoader(args.dataset_dir)
    analyser = EvaluationAnalyser(
        data_loader=data_loader,
        evaluation_dir=args.evaluation_dir,
    )

    if args.output_json:
        output_path = Path(args.output_json)
    else:
        output_path = analyser.choose_run(analyser.discover_runs())

    analyser.analyse(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

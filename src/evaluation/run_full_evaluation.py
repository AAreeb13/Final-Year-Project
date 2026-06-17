from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from src.evaluation.DataLoader import DataLoader
from src.evaluation.EvaluationAnalyser import EvaluationAnalyser
from src.evaluation.EvaluationHarness import EvaluationHarness
from src.evaluation.runSystems import SYSTEM_IDS, register_systems
from src.settings import settings


InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], None]


SYSTEM_DISPLAY_NAMES = {
    "single_agent_system": "Single Agent System",
    "medium_agent_system": "Medium Agent System",
    "multi_agent_system": "Multi-Agent SDLC System",
}


@dataclass(frozen=True)
class TableMetric:
    header: str
    source: str
    unit: str


@dataclass(frozen=True)
class ResultTable:
    section_title: str
    caption: str
    label: str
    metrics: tuple[TableMetric, ...]


RESULT_TABLES: tuple[ResultTable, ...] = (
    ResultTable(
        section_title="Requirements and Design Matching",
        caption="Requirements and design adh results",
        label="tab:req_design_results",
        metrics=(
            TableMetric("Req. Recall", "automated.requirement_recall", "percent"),
            TableMetric("Req. Precision", "automated.requirement_precision", "percent"),
            TableMetric("Constraint Capture", "automated.constraint_capture_rate", "percent"),
            TableMetric("Component Recall", "automated.component_recall", "percent"),
            TableMetric("Module Recall", "automated.module_recall", "percent"),
        ),
    ),
    ResultTable(
        section_title="Architecture and Planning",
        caption="Architecture and planning evaluation results",
        label="tab:architecture_planning_results",
        metrics=(
            TableMetric("Component Precision", "automated.component_precision", "percent"),
            TableMetric("Module Precision", "automated.module_precision", "percent"),
            TableMetric("Dependency Recall", "automated.dependency_recall", "percent"),
            TableMetric(
                "Req.-Component Coverage",
                "manual.architecture.requirement_to_component_coverage",
                "percent",
            ),
        ),
    ),
    ResultTable(
        section_title="Implementation and Testing",
        caption="Implementation and testing results",
        label="tab:implementation_testing_results",
        metrics=(
            TableMetric("Repository Standard", "manual.repository.repository_standard", "score"),
            TableMetric("Build Success", "automated.build_success_rate", "percent"),
            TableMetric("Test Pass Rate", "automated.test_pass_rate", "percent"),
            TableMetric("Test Coverage Score", "automated.test_coverage_score", "percent"),
        ),
    ),
    ResultTable(
        section_title="System Efficiency and Effectiveness",
        caption="System efficiency and effectiveness results",
        label="tab:system_efficiency_results",
        metrics=(
            TableMetric("Completion Rate", "automated.completion_rate", "percent"),
            TableMetric("Iterations", "automated.iteration_count", "count"),
            TableMetric("Tool Calls", "automated.total_tool_calls", "count"),
            TableMetric("Execution Failures", "automated.execution_failure_count", "count"),
            TableMetric("Repair Success", "manual.system_efficiency.repair_success_rate", "percent"),
        ),
    ),
    ResultTable(
        section_title="Software Design Quality",
        caption="Software design quality results",
        label="tab:software_design_quality_results",
        metrics=(
            TableMetric("SOLID", "manual.software_design.solid", "score"),
            TableMetric(
                "Separation of Concerns",
                "manual.software_design.separation_of_concerns",
                "score",
            ),
            TableMetric("Coupling", "manual.software_design.coupling", "score"),
            TableMetric("Cohesion", "manual.software_design.cohesion", "score"),
            TableMetric("Maintainability", "manual.software_design.maintainability", "score"),
        ),
    ),
)


def run_all_systems(
    *,
    dataset_dir: str | Path,
    evaluation_dir: str | Path,
    selected_system: str = "all",
    datapoint: str | None = None,
    all_datapoints: bool = True,
    human_approval: bool = False,
    debug_structured_output: bool = False,
    verbose: bool = False,
) -> Path:
    data_loader = DataLoader(str(Path(dataset_dir).expanduser()))
    harness = EvaluationHarness(
        data_loader=data_loader,
        output_dir=Path(evaluation_dir).expanduser(),
    )
    register_systems(harness, selected_system)

    if datapoint and not all_datapoints:
        if selected_system == "all":
            harness.run_datapoint_with_all_systems(
                datapoint_id=datapoint,
                human_approval=human_approval,
                debug_structured_output=debug_structured_output,
                verbose=verbose,
            )
        else:
            harness.run_datapoint_with_system(
                datapoint_id=datapoint,
                human_approval=human_approval,
                system_id=selected_system,
                debug_structured_output=debug_structured_output,
                verbose=verbose,
            )
    elif selected_system == "all":
        harness.run_all_datapoints_with_all_systems(
            human_approval=human_approval,
            debug_structured_output=debug_structured_output,
            verbose=verbose,
        )
    else:
        harness.run_all_datapoints_with_system(
            system_id=selected_system,
            human_approval=human_approval,
            debug_structured_output=debug_structured_output,
            verbose=verbose,
        )

    if harness.eval_dir is None:
        raise RuntimeError("No evaluation directory was created. Did any systems run?")
    return harness.eval_dir


def analyse_evaluation_dir(
    *,
    dataset_dir: str | Path,
    eval_dir: str | Path,
    overwrite_analysis: bool = False,
    input_func: InputFunc = input,
    output: OutputFunc = print,
) -> list[Path]:
    data_loader = DataLoader(str(Path(dataset_dir).expanduser()))
    analyser = EvaluationAnalyser(data_loader=data_loader, evaluation_dir=Path(eval_dir), output=output)
    saved_paths: list[Path] = []

    for output_json in discover_output_json(eval_dir):
        existing_evaluation = latest_evaluation_path(output_json.parent)
        if existing_evaluation is not None and not overwrite_analysis:
            output(f"Skipping already analysed run: {output_json}")
            continue
        output(f"Analysing run: {output_json}")
        saved_paths.append(analyser.analyse(output_json, input_func=input_func))

    return saved_paths


def aggregate_evaluation_dir(
    eval_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    eval_path = Path(eval_dir).expanduser().resolve()
    summaries = load_evaluation_summaries(eval_path)
    markdown = render_markdown_summary(summaries)
    latex = render_latex_summary(summaries)

    destination = Path(output_dir).expanduser().resolve() if output_dir else eval_path
    destination.mkdir(parents=True, exist_ok=True)
    markdown_path = destination / "summary.md"
    latex_path = destination / "summary.tex"
    markdown_path.write_text(markdown, encoding="utf-8")
    latex_path.write_text(latex, encoding="utf-8")
    return {"markdown": markdown_path, "latex": latex_path}


def load_evaluation_summaries(eval_dir: str | Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for output_json in discover_output_json(eval_dir):
        evaluation_path = latest_evaluation_path(output_json.parent)
        if evaluation_path is None:
            continue
        with evaluation_path.open("r", encoding="utf-8") as file:
            summaries.append(json.load(file))
    return summaries


def discover_output_json(eval_dir: str | Path) -> list[Path]:
    return sorted(Path(eval_dir).expanduser().resolve().glob("*/*/*/output.json"))


def resolve_eval_dir(evaluation_dir: str | Path, eval_id: str = "latest") -> Path:
    root = Path(evaluation_dir).expanduser().resolve()
    if eval_id == "latest":
        latest = latest_eval_dir(root)
        if latest is None:
            raise ValueError(f"No eval_* directories found in {root}")
        return latest

    if eval_id.startswith("eval_"):
        candidate = root / eval_id
    else:
        candidate = root / f"eval_{int(eval_id):03d}"

    if not candidate.exists() or not candidate.is_dir():
        raise ValueError(f"Evaluation directory does not exist: {candidate}")
    return candidate


def latest_eval_dir(evaluation_dir: str | Path) -> Path | None:
    root = Path(evaluation_dir).expanduser().resolve()
    eval_dirs = [
        path
        for path in root.glob("eval_*")
        if path.is_dir() and _eval_number(path.name) is not None
    ]
    if not eval_dirs:
        return None
    return max(eval_dirs, key=lambda path: _eval_number(path.name) or 0)


def latest_evaluation_path(run_dir: str | Path) -> Path | None:
    run_path = Path(run_dir).expanduser().resolve()
    candidates = [
        path
        for path in run_path.glob("evaluation*.json")
        if _evaluation_number(path.name) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: _evaluation_number(path.name) or 0)


def render_markdown_summary(summaries: Sequence[dict[str, Any]]) -> str:
    lines = ["# Evaluation Summary", ""]
    for table in RESULT_TABLES:
        lines.append(f"## {table.section_title}")
        lines.append("")
        headers = ["System", *(metric.header for metric in table.metrics)]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for system_id in SYSTEM_IDS:
            values = aggregate_table_row(summaries, system_id, table.metrics)
            row = [SYSTEM_DISPLAY_NAMES.get(system_id, system_id)]
            row.extend(_format_value(values[metric.header], metric.unit) for metric in table.metrics)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_latex_summary(summaries: Sequence[dict[str, Any]]) -> str:
    sections = [
        "\\section{Results 1: Software Engineering Design Quality}",
    ]
    for table in RESULT_TABLES:
        column_spec = "l" + ("c" * len(table.metrics))
        headers = ["\\textbf{System}", *(_latex_bold(metric.header) for metric in table.metrics)]
        rows = [
            "\\subsection{" + _latex_escape(table.section_title) + "}",
            "",
            "\\begin{table}[H]",
            "\\centering",
            "\\resizebox{\\textwidth}{!}{%",
            f"\\begin{{tabular}}{{{column_spec}}}",
            "\\hline",
            " & ".join(headers) + " \\\\",
            "\\hline",
        ]
        for system_id in SYSTEM_IDS:
            values = aggregate_table_row(summaries, system_id, table.metrics)
            cells = [_latex_escape(SYSTEM_DISPLAY_NAMES.get(system_id, system_id))]
            cells.extend(
                _latex_escape(_format_value(values[metric.header], metric.unit))
                for metric in table.metrics
            )
            rows.append(" & ".join(cells) + " \\\\")
        rows.extend(
            [
                "\\hline",
                "\\end{tabular}%",
                "}",
                "\\caption{" + _latex_escape(table.caption) + "}",
                "\\label{" + table.label + "}",
                "\\end{table}",
            ]
        )
        sections.append("\n".join(rows))
    return "\n\n".join(sections).rstrip() + "\n"


def aggregate_table_row(
    summaries: Sequence[dict[str, Any]],
    system_id: str,
    metrics: Iterable[TableMetric],
) -> dict[str, float | None]:
    system_summaries = [
        summary
        for summary in summaries
        if summary.get("metadata", {}).get("system_id") == system_id
    ]
    return {
        metric.header: _average(
            _metric_value(summary, metric.source)
            for summary in system_summaries
        )
        for metric in metrics
    }


def _metric_value(summary: dict[str, Any], source: str) -> float | None:
    source_kind, key = source.split(".", 1)
    if source_kind == "automated":
        value = summary.get("automated_metrics", {}).get(key)
    elif source_kind == "manual":
        value = summary.get("manual_metrics", {}).get(key, {}).get("score")
    else:
        raise ValueError(f"Unknown metric source: {source}")
    return _numeric(value)


def _average(values: Iterable[float | None]) -> float | None:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return round(sum(numeric_values) / len(numeric_values), 2)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: float | None, unit: str) -> str:
    if value is None:
        return "N/A"
    rendered = _format_number(value)
    if unit == "percent":
        return f"{rendered}%"
    if unit == "score":
        return f"{rendered}/10"
    return rendered


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _eval_number(name: str) -> int | None:
    if not name.startswith("eval_"):
        return None
    try:
        return int(name.removeprefix("eval_"))
    except ValueError:
        return None


def _evaluation_number(name: str) -> int | None:
    if name == "evaluation.json":
        return 1
    if not name.startswith("evaluation_") or not name.endswith(".json"):
        return None
    try:
        return int(name.removeprefix("evaluation_").removesuffix(".json"))
    except ValueError:
        return None


def _latex_bold(value: str) -> str:
    return "\\textbf{" + _latex_escape(value) + "}"


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _default_dataset_dir() -> Path:
    if settings.DATASET_DIRECTORY is None:
        raise ValueError("DATASET_DIRECTORY must be set or --dataset-dir must be provided.")
    return Path(settings.DATASET_DIRECTORY).expanduser()


def _default_evaluation_dir() -> Path:
    if settings.EVALUATION_DIRECTORY is None:
        raise ValueError("EVALUATION_DIRECTORY must be set or --evaluation-dir must be provided.")
    return Path(settings.EVALUATION_DIRECTORY).expanduser()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run, analyse, and aggregate full evaluation results.")
    parser.add_argument("command", choices=("run", "analyse", "aggregate", "all"))
    parser.add_argument("--dataset-dir", help="Path to the Revolutionising-SWE-dataset folder.")
    parser.add_argument("--evaluation-dir", help="Path where eval_* directories are stored.")
    parser.add_argument("--eval-id", default="latest", help="Evaluation id for analyse/aggregate, e.g. latest, 1, eval_001.")
    parser.add_argument("--system", choices=("all", *SYSTEM_IDS), default="all")
    parser.add_argument("--datapoint", help="Run a single datapoint. By default every datapoint is run.")
    parser.add_argument("--all-datapoints", action="store_true", help="Run every datapoint, overriding --datapoint.")
    parser.add_argument("--overwrite-analysis", action="store_true", help="Analyse runs even if evaluation.json already exists.")
    parser.add_argument("--debug-structured-output", action="store_true")
    parser.add_argument("--human-approval", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print compact graph stage and command progress.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(args.dataset_dir).expanduser() if args.dataset_dir else _default_dataset_dir()
    evaluation_dir = (
        Path(args.evaluation_dir).expanduser() if args.evaluation_dir else _default_evaluation_dir()
    )

    eval_dir: Path | None = None
    if args.command in {"run", "all"}:
        eval_dir = run_all_systems(
            dataset_dir=dataset_dir,
            evaluation_dir=evaluation_dir,
            selected_system=args.system,
            datapoint=args.datapoint,
            all_datapoints=args.all_datapoints or args.datapoint is None,
            human_approval=args.human_approval,
            debug_structured_output=args.debug_structured_output,
            verbose=args.verbose,
        )
        print(f"Run outputs saved under: {eval_dir}")

    if args.command in {"analyse", "aggregate"}:
        eval_dir = resolve_eval_dir(evaluation_dir, args.eval_id)

    if args.command in {"analyse", "all"}:
        if eval_dir is None:
            raise RuntimeError("No evaluation directory selected for analysis.")
        analyse_evaluation_dir(
            dataset_dir=dataset_dir,
            eval_dir=eval_dir,
            overwrite_analysis=args.overwrite_analysis,
        )

    if args.command in {"aggregate", "all"}:
        if eval_dir is None:
            raise RuntimeError("No evaluation directory selected for aggregation.")
        output_paths = aggregate_evaluation_dir(eval_dir)
        print(f"Wrote Markdown summary to: {output_paths['markdown']}")
        print(f"Wrote LaTeX summary to: {output_paths['latex']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

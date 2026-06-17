import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.DataLoader import DataLoader
from src.evaluation.EvaluationAnalyser import (
    EvaluationAnalyser,
    MANUAL_METRICS,
    _render_side_by_side,
)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def create_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "Revolutionising-SWE-dataset"
    project_dir = dataset_root / "dataset" / "snake_game"
    project_dir.mkdir(parents=True)

    write_text(
        project_dir / "project_info.yaml",
        """
project_id: snake_game
project_prompt: Build a snake game.
project_type: game
difficulty: low
""",
    )
    write_text(
        project_dir / "requirements.yaml",
        """
requirements:
  functional:
    - The snake can be moved by the user
    - Food appears at random locations
  non_functional:
    - The game is written in Python
""",
    )
    write_text(
        project_dir / "design.yaml",
        """
high_level_design:
  style: Modular
  components:
    - name: Snake Game
      responsibility: Runs the game loop
  relationships:
    - source: Input Handler
      target: Snake
      relationship_type: sends movement commands to
  technologies:
    - Python
""",
    )
    write_text(
        project_dir / "modules.yaml",
        """
modules:
  - name: Snake
    component: Snake Game
    type: class
    responsibilities:
      - Represent the snake state
    dependencies: []
""",
    )
    write_text(
        project_dir / "graph_dependency_spec.yaml",
        """
graph_dependencies:
  - component: Snake Game
    Relationships:
      - source: Input Handler
        target: Snake
        relationship_type: sends movement commands to
""",
    )
    write_text(
        project_dir / "repository_structure.yaml",
        """
repository_structure:
  - src/
  - src/snake.py
""",
    )
    write_text(
        project_dir / "implementation_plan.yaml",
        """
implementation_plan:
  - step_id: 1
    description: Implement Snake
""",
    )

    return dataset_root


def create_output_json(tmp_path: Path) -> Path:
    output_path = (
        tmp_path
        / "agent_evaluation"
        / "eval_001"
        / "snake_game"
        / "single_agent_system"
        / "run_single_agent_system"
        / "output.json"
    )
    output_path.parent.mkdir(parents=True)
    output = {
        "project_id": None,
        "system_name": "Single Agent",
        "status": "completed",
        "requirements": {
            "functional": ["The snake can be moved by the user"],
            "non_functional": ["The game is written in Python"],
        },
        "high_level_design": {
            "style": "Modular",
            "components": [{"name": "Snake Game"}],
            "relationships": [
                {
                    "source": "Input Handler",
                    "target": "Snake",
                    "relationship_type": "sends movement commands to",
                }
            ],
            "technologies": ["Python"],
        },
        "modules": [
            {
                "name": "Snake",
                "component": "Snake Game",
                "type": "class",
                "responsibilities": ["Represent the snake state"],
            }
        ],
        "module_dependency_graph": {
            "nodes": ["Input Handler", "Snake"],
            "edges": [{"source": "Input Handler", "target": "Snake"}],
        },
        "repository_structure": {
            "directories": [
                {
                    "name": "src",
                    "parent": None,
                    "children": [],
                    "files": [{"name": "snake.py", "modules": []}],
                }
            ]
        },
        "test_plan": {"unit_tests": ["test snake movement"], "commands": ["pytest"]},
        "generated_files": [{"path": "src/snake.py", "content": "class Snake: pass"}],
        "tool_calls": [{"tool_name": "write_file", "success": True}],
    }
    output_path.write_text(json.dumps(output), encoding="utf-8")
    return output_path


def create_analyser(tmp_path: Path) -> tuple[EvaluationAnalyser, Path, Path]:
    dataset_root = create_dataset(tmp_path)
    output_path = create_output_json(tmp_path)
    loader = DataLoader(str(dataset_root))
    analyser = EvaluationAnalyser(loader, output=lambda _message: None)
    return analyser, dataset_root, output_path


def test_load_comparison_inputs_matches_datapoint_from_run_path(tmp_path: Path):
    analyser, _, output_path = create_analyser(tmp_path)

    project, result = analyser.load_comparison_inputs(output_path)

    assert project.project_id == "snake_game"
    assert result.system_name == "Single Agent"


def test_discover_runs_finds_output_json_files(tmp_path: Path):
    analyser, _, output_path = create_analyser(tmp_path)
    analyser.evaluation_dir = output_path.parents[4]

    assert analyser.discover_runs() == [output_path.resolve()]


def test_side_by_side_rendering_handles_missing_sides():
    both = _render_side_by_side("Example", ["reference"], ["output"])
    output_only = _render_side_by_side("Example", [], ["output"])
    empty = _render_side_by_side("Example", [], [])

    assert "Dataset" in both
    assert "Output" in both
    assert "Dataset" not in output_only
    assert "Output" in output_only
    assert empty == ""


def test_prompt_score_validates_general_and_percent_metrics(tmp_path: Path):
    analyser, _, _ = create_analyser(tmp_path)
    general_metric = next(metric for metric in MANUAL_METRICS if metric.unit == "score")
    percent_metric = MANUAL_METRICS[0]

    general_inputs = iter(["11", "9.5"])
    percent_inputs = iter(["101", "75"])

    assert (
        analyser.prompt_score(
            general_metric,
            input_func=lambda _prompt: next(general_inputs),
        )
        == 9.5
    )
    assert (
        analyser.prompt_score(
            percent_metric,
            input_func=lambda _prompt: next(percent_inputs),
        )
        == 75
    )


def test_manual_metrics_include_quantitative_metric_prompts():
    metric_ids = {metric.metric_id for metric in MANUAL_METRICS}

    assert {
        "requirement_extraction.requirements_precision",
        "requirement_extraction.constraint_capture_rate",
        "architecture.component_recall",
        "architecture.component_precision",
        "architecture.module_recall",
        "architecture.module_precision",
        "architecture.dependency_recall",
        "architecture.requirement_to_component_coverage",
        "repository.repository_structure_accuracy",
        "repository.repository_standard",
        "execution.build_success_rate",
        "execution.test_pass_rate",
        "test_coverage.test_coverage_score",
        "system_efficiency.completion_rate",
        "system_efficiency.iteration_count",
        "system_efficiency.tool_call_count",
        "system_efficiency.execution_failure_count",
        "system_efficiency.repair_success_rate",
        "system_efficiency.trace_consistency",
        "software_design.solid_violations",
    }.issubset(metric_ids)


def test_manual_score_metrics_are_out_of_ten():
    score_metrics = [metric for metric in MANUAL_METRICS if metric.unit == "score"]

    assert score_metrics
    assert all(metric.max_score == 10 for metric in score_metrics)


def test_next_evaluation_path_versions_existing_files(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "evaluation.json").write_text("{}", encoding="utf-8")
    (run_dir / "evaluation_002.json").write_text("{}", encoding="utf-8")

    assert EvaluationAnalyser.next_evaluation_path(run_dir).name == "evaluation_003.json"


def test_build_and_save_evaluation_includes_automated_metrics(tmp_path: Path):
    analyser, _, output_path = create_analyser(tmp_path)
    project, result = analyser.load_comparison_inputs(output_path)
    manual_metrics = {
        "requirement_extraction.requirements_recall": {
            "score": 80,
            "max_score": 100,
            "unit": "percent",
            "qualitative_feedback": "Mostly complete.",
        }
    }

    evaluation = analyser.build_evaluation_summary(
        output_path=output_path,
        project=project,
        result=result,
        manual_metrics=manual_metrics,
    )
    save_path = analyser.save_evaluation(output_path, evaluation)
    saved = json.loads(save_path.read_text(encoding="utf-8"))

    assert save_path.name == "evaluation.json"
    assert saved["metadata"]["datapoint_id"] == "snake_game"
    assert saved["automated_metrics"]["total_requirements"] == 3
    assert saved["automated_metrics"]["implemented_requirements"] == 2
    assert saved["automated_metrics"]["requirement_recall"] == 66.67
    assert saved["automated_metrics"]["test_pass_rate"] == 0
    assert saved["automated_metrics"]["tool_call_success_rate"] == 100
    assert saved["manual_metrics"] == manual_metrics


def test_format_automated_metrics_shows_thesis_metric_summary(tmp_path: Path):
    analyser, _, output_path = create_analyser(tmp_path)
    project, result = analyser.load_comparison_inputs(output_path)
    metrics = analyser.build_evaluation_summary(
        output_path=output_path,
        project=project,
        result=result,
        manual_metrics={},
    )["automated_metrics"]

    rendered = analyser.format_automated_metrics(metrics)

    assert "Automated Metrics" in rendered
    assert "Requirement Recall: 66.67%" in rendered
    assert "Repository Structure Accuracy: 100%" in rendered


def test_data_loader_retrieve_full_datapoint_uses_loader_dataset_path(tmp_path: Path):
    dataset_root = create_dataset(tmp_path)
    loader = DataLoader(str(dataset_root))

    project = loader.retrieve_full_datapoint("snake_game")

    assert project.project_id == "snake_game"
    assert project.requirements.functional == [
        "The snake can be moved by the user",
        "Food appears at random locations",
    ]

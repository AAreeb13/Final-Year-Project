from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.AgentSystem import AgentSystemRunner
from src.evaluation.DataLoader import DataLoader
from src.settings import settings

class EvaluationHarness:
    def __init__(self, data_loader: DataLoader, output_dir: str | Path):
        self.systems: list[AgentSystemRunner] = []
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)

        # eval_001, eval_002, etc. is created lazily when the first result is saved.
        self.eval_dir: Path | None = None

        print("EvaluationHarness initialized with output directory:", self.output_dir)

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error creating output directory {self.output_dir}: {e}")

        if not self.systems:
            print(
                "  Warning: No systems have been registered with the EvaluationHarness yet.\n"
                "  Please register systems before running evaluations."
            )

    def run_datapoint_with_system(
        self,
        datapoint_id: str,
        human_approval: bool = True,
        system_id: int | str | None = None,
    ):
        print(f"========= Running system {system_id} for datapoint {datapoint_id}.")

        datapoint = self.data_loader.load_datapoint(datapoint_id)
        problem_statement = datapoint.get("project_prompt", "")

        run_config = {
            "human_approval": human_approval,
            "allow_tool_execution": False,
            "debug_structured_output": False,
        }

        system = self._get_system(system_id)

        result, run_id = system.run(
            prompt=problem_statement,
            run_config=run_config,
        )

        try:
            self.save_run_artifact(
                result=result,
                run_id=run_id,
                datapoint_id=datapoint_id,
                system_id=system.system_id,
            )
        except Exception as e:
            print(
                f"Error saving result for datapoint {datapoint_id} "
                f"with run_id {run_id}: {e}"
            )
        finally:
            print(f"===== Completed running datapoint {datapoint_id} with all systems. =====")


    def run_datapoint_with_all_systems(
        self,
        datapoint_id: str,
        human_approval: bool = True,
    ):
        print()
        print(f"===== Running datapoint {datapoint_id} with registered systems. =====")

        for system_index in range(len(self.systems)):
            self.run_datapoint_with_system(
                datapoint_id,
                human_approval=human_approval,
                system_id=system_index,
            )


    def run_all_datapoints_with_all_systems(self, human_approval: bool = True):
        datapoint_ids = self.data_loader.list_datapoints(id_only=True)

        for datapoint_id in datapoint_ids:
            self.run_datapoint_with_all_systems(
                datapoint_id,
                human_approval=human_approval,
            )

    def run_all_datapoints_with_system(
        self,
        system_id: int | str = 0,
        human_approval: bool = True,
    ):
        print("Running all datapoints with system:", system_id)
        datapoint_ids = self.data_loader.list_datapoints(id_only=True)

        for datapoint_id in datapoint_ids:
            self.run_datapoint_with_system(
                datapoint_id,
                human_approval=human_approval,
                system_id=system_id,
            )

    def save_run_artifact(
        self,
        result: dict[str, Any] | Any,
        run_id: str,
        datapoint_id: str | None = None,
        system_id: str | None = None,
    ) -> Path:
        """
        evaluation_results/
          eval_001/
            datapoint_id/
              system_id/
                run_id/
                  output.json
        """

        eval_dir = self._get_or_create_eval_dir()

        safe_datapoint_id = datapoint_id or "unknown_datapoint"
        safe_system_id = system_id or "unknown_system"
        safe_run_id = run_id or "unknown_run"

        run_artifact_dir = (
            eval_dir
            / safe_datapoint_id
            / safe_system_id
            / safe_run_id
        )

        run_artifact_dir.mkdir(parents=True, exist_ok=True)

        output_path = run_artifact_dir / "output.json"

        serialisable_result = self._make_json_serialisable(result)

        output_path.write_text(
            json.dumps(serialisable_result, indent=2),
            encoding="utf-8",
        )

        print(f"Saved run artifact to: {output_path}")

        return output_path

    def _register_system(self, system: AgentSystemRunner):
        if not isinstance(system, AgentSystemRunner):
            raise ValueError(
                f"System must be an instance of AgentSystem, got {type(system)}"
            )

        if any(s.system_id == system.system_id for s in self.systems):
            print(
                f"Warning: System with ID {system.system_id} is already registered. "
                "Skipping registration."
            )
            return

        self.systems.append(system)

    def _get_system(self, system_id: int | str | None) -> AgentSystemRunner:
        if not self.systems:
            raise ValueError(
                "No systems have been registered with the EvaluationHarness."
            )

        if system_id is None:
            return self.systems[0]

        if isinstance(system_id, int):
            try:
                return self.systems[system_id]
            except IndexError as exc:
                raise ValueError(f"No system exists at index {system_id}") from exc

        if isinstance(system_id, str):
            for system in self.systems:
                if system.system_id == system_id:
                    return system

            raise ValueError(f"No system registered with system_id '{system_id}'")

        raise TypeError(
            f"system_id must be int, str, or None. Got {type(system_id)}"
        )

    def _get_or_create_eval_dir(self) -> Path:
        if self.eval_dir is not None:
            return self.eval_dir

        self.eval_dir = self._create_next_eval_dir()
        return self.eval_dir

    def _create_next_eval_dir(self) -> Path:
        existing_eval_dirs = [
            path
            for path in self.output_dir.iterdir()
            if path.is_dir() and path.name.startswith("eval_")
        ]

        existing_numbers = []

        for path in existing_eval_dirs:
            try:
                number = int(path.name.replace("eval_", ""))
                existing_numbers.append(number)
            except ValueError:
                continue

        next_number = max(existing_numbers, default=0) + 1
        eval_dir = self.output_dir / f"eval_{next_number:03d}"
        eval_dir.mkdir(parents=True, exist_ok=False)

        print(f"Created evaluation directory: {eval_dir}")

        return eval_dir

    def _make_json_serialisable(self, result: Any) -> Any:
        """
        Supports plain dicts and Pydantic models.
        """

        if hasattr(result, "model_dump"):
            return result.model_dump()

        if isinstance(result, dict):
            return result

        return {"result": str(result)}
    
if __name__ == "__main__":
    from pathlib import Path

    from src.settings import settings
    from src.evaluation.DataLoader import DataLoader

    if settings.EVALUATION_DIRECTORY is None:
        raise ValueError(
            "EVALUATION_DIRECTORY must be set in the .env file to run the evaluation harness."
        )

    if settings.DATASET_DIRECTORY is None:
        raise ValueError(
            "DATASET_DIRECTORY must be set in the .env file to run the evaluation harness."
        )

    evaluation_directory = Path(settings.EVALUATION_DIRECTORY).expanduser().resolve()
    dataset_directory = Path(settings.DATASET_DIRECTORY).expanduser().resolve()

    print("Starting EvaluationHarness")
    print(f"Dataset directory: {dataset_directory}")
    print(f"Evaluation directory: {evaluation_directory}")

    data_loader = DataLoader(str(dataset_directory))

    harness = EvaluationHarness(
        data_loader=data_loader,
        output_dir=evaluation_directory,
    )
    from src.evaluation.runners import SingleAgentSystemRunner as SASrunner
    system1 = SASrunner(description="Single agent system using the master prompt")

    harness._register_system(system1)
    print("Available datapoints:")
    datapoints = data_loader.list_datapoints()

    for datapoint in datapoints:
        print(f"- {datapoint.get('project_id')}")

    print()
    print("EvaluationHarness setup completed.")
    print("No systems have been registered yet, so no evaluation run was started.")

    for system in harness.systems:
        system.display_architecture()

    harness.run_all_datapoints_with_all_systems(human_approval=False)

import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.EvaluationHarness import EvaluationHarness
from src.evaluation.DataLoader import DataLoader
from src.evaluation.AgentSystem import AgentSystemRunner
# Create a mock DataLoader


class MockDataLoader(DataLoader):
    def __init__(self, data_path):
        self.data_path = data_path
    def load_datapoint(self, datapoint_id):
        return {"project_prompt": "Test prompt for datapoint " + datapoint_id,
                    "project_id": datapoint_id}
    
    def get_all_datapoint_ids(self):
        return ["datapoint1", "datapoint2"]
    
class MockSystem(AgentSystemRunner):
    def __init__(self, system_id):
        self.system_id = system_id
        self.reset_reasons = []
    def run(self, prompt, run_config):
        return {"result": f"Result from system {self.system_id} for prompt: {prompt}"}, f"run_{self.system_id}"
    def reset_system(self, *, reason="", verbose=False):
        self.reset_reasons.append((reason, verbose))
    def display_architecture(self):
        print(f"Architecture for system <{self.system_id}>")

def test_evaluation_harness_initialization(tmp_path):

    data_loader = MockDataLoader(tmp_path)  # Using tmp_path as a dummy path for initialization
    output_dir = tmp_path / "evaluation_results"
    harness = EvaluationHarness(data_loader, output_dir)
    
    # check if output directory is created
    assert output_dir.exists() and output_dir.is_dir(), "Output directory was not created successfully."
    # delete dir
    output_dir.rmdir()
    
    assert harness.data_loader == data_loader
    assert harness.output_dir == output_dir

def test_register_mock_system(tmp_path):
    data_loader = MockDataLoader(tmp_path)
    output_dir = tmp_path / "evaluation_results"
    harness = EvaluationHarness(data_loader, output_dir)
    test_id = "mock_system_1"
    mock_system = MockSystem(test_id)
    harness._register_system(mock_system)
    assert len(harness.systems) == 1
    assert harness.systems[0].system_id == test_id


def test_save_run_artifact_creates_eval_directory(tmp_path):
    data_loader = MockDataLoader(tmp_path)
    output_dir = tmp_path / "evaluation_results"
    harness = EvaluationHarness(data_loader, output_dir)

    result = {"result": "hello"}
    output_path = harness.save_run_artifact(
        result=result,
        run_id="run_001",
        datapoint_id="datapoint1",
        system_id="mock_system_1",
    )

    assert output_path.exists()
    assert output_path.name == "output.json"
    assert "eval_001" in str(output_path)

    saved = output_path.read_text(encoding="utf-8")
    assert "hello" in saved


def test_harness_resets_system_before_and_after_run(tmp_path):
    data_loader = MockDataLoader(tmp_path)
    harness = EvaluationHarness(data_loader, tmp_path / "evaluation_results")
    system = MockSystem("mock_system_1")
    harness._register_system(system)

    harness.run_datapoint_with_system(
        "datapoint1",
        human_approval=False,
        system_id="mock_system_1",
        verbose=True,
    )

    assert system.reset_reasons == [
        ("before datapoint datapoint1", True),
        ("after datapoint datapoint1", True),
    ]

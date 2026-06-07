from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import yaml

from src.evaluation.helper import load_yaml_file


def test_load_yaml_file_valid(tmp_path: Path):
    test_yaml_path = tmp_path / "test.yaml"
    test_data = {"key": "value"}

    with test_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(test_data, f)

    loaded_data = load_yaml_file(test_yaml_path)

    assert loaded_data == test_data


def test_load_yaml_file_not_found():
    non_existent_path = Path("non_existentauwbhdiuhw2i.yaml")

    with pytest.raises(ValueError, match="YAML file not found"):
        load_yaml_file(non_existent_path)

import json

import docker
import pytest

from src.settings import settings
from src.tools.docker_code_execution.tool import (
    DEFAULT_IMAGE,
    run_in_container,
    stop_container,

)


@pytest.fixture(scope="module")
def docker_client():
    try:
        client = docker.from_env()
        client.ping()
        client.images.get(DEFAULT_IMAGE)
    except Exception as error:
        pytest.skip(f"Docker test image is not available: {error}")

    return client


@pytest.fixture
def temp_workplace(tmp_path, monkeypatch):
    workplace = tmp_path / "workplace"
    workplace.mkdir()
    monkeypatch.setattr(settings, "WORKPLACE_FOLDER", str(workplace))
    return workplace


def invoke_container(command: list[str], timeout_s: int = 30) -> dict:
    raw_output = run_in_container.invoke(
        {
            "command": command,
            "timeout_s": timeout_s,
        }
    )
    # stop container
    stop_container.invoke({})
    
    return json.loads(raw_output)


def test_run_in_container_returns_success_schema(docker_client, temp_workplace):
    (temp_workplace / "hello.py").write_text('print("hello from temp workplace")\n')

    output = invoke_container(["python", "hello.py"])

    assert output == {
        "status": "success",
        "exit_code": 0,
        "stdout": "hello from temp workplace\n",
        "stderr": "",
        "message": "Command completed successfully.",
    }


def test_run_in_container_uses_temporary_workplace(docker_client, temp_workplace):
    output = invoke_container(
        [
            "python",
            "-c",
            "from pathlib import Path; Path('created_by_container.txt').write_text('container write')",
        ]
    )

    assert output["status"] == "success"
    assert (temp_workplace / "created_by_container.txt").read_text() == "container write"


def test_run_in_container_returns_failure_schema(docker_client, temp_workplace):
    output = invoke_container(
        [
            "python",
            "-c",
            "import sys; print('stdout text'); print('stderr text', file=sys.stderr); sys.exit(7)",
        ]
    )

    assert output["status"] == "error"
    assert output["exit_code"] == 7
    assert output["stdout"] == "stdout text\n"
    assert output["stderr"] == "stderr text\n"
    assert output["message"] == "Command failed with exit code 7."


def test_run_in_container_times_out(docker_client, temp_workplace):
    output = invoke_container(
        ["python", "-c", "import time; time.sleep(2)"],
        timeout_s=1,
    )

    assert output["status"] == "error"
    assert output["exit_code"] == 124
    assert output["message"] == "Command timed out."


def test_stop_container_when_no_container_is_running(docker_client):
    # This should not raise an exception even if no container is running
    try:
        output = stop_container.invoke({})
        # assert output["status"] == "success"
        assert type(output) == str, f"Expected output to be a string, got {type(output)}"
        output = json.loads(output)
        assert int(output["exit_code"]) == 0, f"Expected exit code 0 when stopping container, got {output}"

        assert output["status"] == "success", f"Expected status 'success' when stopping container, got {output['status']}"
    except Exception as e:
        pytest.fail(f"stop_container raised an exception when no container was running: {e}")

def test_run_in_container_with_a_folder_in_workplace(docker_client, temp_workplace):
    # Create a folder and a file inside it
    folder = temp_workplace / "test_folder"
    folder.mkdir()
    (folder / "test_file.txt").write_text("This is a test file.")

    output = invoke_container(
        [
            "python",
            "-c",
            "from pathlib import Path; print(Path('test_folder/test_file.txt').read_text())",
        ]
    )

    assert output["status"] == "success"
    assert output["stdout"] == "This is a test file.\n"
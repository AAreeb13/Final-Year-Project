import os
from typing import Sequence

import docker
from docker.errors import DockerException
from docker.models.containers import Container
from src.settings import settings

DEFAULT_IMAGE = "agent-python-git:latest"
DEFAULT_WORKSPACE_BIND = "/workspace"


def _get_client() -> docker.DockerClient:
    try:
        client = docker.from_env()
        client.ping()
        return client
    except (DockerException, FileNotFoundError, OSError) as error:
        raise RuntimeError(
            "Docker is not available. Start the Docker daemon or make sure the Docker socket is mounted "
            "before running this tool."
        ) from error

def _format_exec_output(exit_code: int, output: bytes) -> str:
    text_output = output.decode(errors="replace")
    if exit_code == 0:
        return text_output

    return f"Command failed with exit code {exit_code}.\n{text_output}"


def start_container(
    workspace_path: str,
    image: str = DEFAULT_IMAGE,
    memory_limit: str = "256m",
    container_name: str | None = None,
) -> str:
    # print("WOrkspace path:", workspace_path)
    print("Starting Docker container...")

    client = _get_client()
    # print("Got Docker client.")
    absolute_workspace_path = os.path.abspath(workspace_path)
    print(f"absolute_workspace_path: {absolute_workspace_path}")
    container: Container = client.containers.run(
        image=image,
        command=["sleep", "infinity"],
        detach=True,
        name=container_name,
        volumes={
            absolute_workspace_path: {
                "bind": DEFAULT_WORKSPACE_BIND,
                "mode": "rw",
            }
        },
        working_dir=DEFAULT_WORKSPACE_BIND,
        network_disabled=True,
        mem_limit=memory_limit,
    )

    return container.id


def run_command_in_container(container_id: str, command: Sequence[str]) -> str:
    client = _get_client()
    container = client.containers.get(container_id)

    exit_code, output = container.exec_run(
        cmd=list(command),
        workdir=DEFAULT_WORKSPACE_BIND,
    )

    return _format_exec_output(exit_code, output)


def close_container(container_id: str) -> None:
    client = _get_client()
    container = client.containers.get(container_id)
    container.stop()
    container.remove()


def run_in_container(command: list[str], workspace_path: str) -> str:
    
    container_id = start_container(workspace_path)
    try:
        return run_command_in_container(container_id, command)
    finally:
        close_container(container_id)


if __name__ == "__main__":
    print("Testing Docker Code Execution...")

    folder = settings.WORKPLACE_FOLDER
    print("Initializing folder...")
    if not os.path.exists(folder):
        os.makedirs(folder)

    test_file = """print("Hello, World!")
"""
    # check if the file already exists, if not create it
    if not os.path.exists(f"{folder}/test.py"):
        print("Creating test file...")
        with open(f"{folder}/test.py", "w") as f:
            f.write(test_file)


    command = ["python", "test.py"]

    try:
        output = run_in_container(command, folder)
        print("Command output:")
        print(output)
    except Exception as error:
        print(f"Error: {error}")
    
import os
from typing import Sequence

import docker
from docker.errors import DockerException, APIError, NotFound
from docker.models.containers import Container
from langchain_core.tools import tool
from pydantic import BaseModel

from src.settings import settings
from src.tools.tool_schemas import RunInContainerSchema


DEFAULT_IMAGE = "agent-python-git:latest"
DEFAULT_WORKSPACE_BIND = "/workspace"
DEFAULT_CONTAINER_ENV = {
    "HOME": DEFAULT_WORKSPACE_BIND,
    "PIP_CACHE_DIR": f"{DEFAULT_WORKSPACE_BIND}/.cache/pip",
}

class DockerCommandOutputSchema(BaseModel):
    """
    Output schema for Docker command execution.
    """
    status: str
    exit_code: int | None
    stdout: str
    stderr: str
    message: str

@tool
def stop_container() -> str:
    """Stop and remove the project Docker container."""
    container_name = _get_container_name()

    try:
        _close_container(container_name)
        return _build_output(
            status="success",
            exit_code=0,
            message=f"Container stopped and removed: {container_name}",
        )

    except NotFound:
        return _build_output(
            status="success",
            exit_code=0,
            message=f"Container was not running or did not exist: {container_name}",
        )

    except Exception as error:
        return _build_output(
            status="error",
            exit_code=None,
            stderr=str(error),
            message="Failed to stop the project container.",
        )

@tool(args_schema=RunInContainerSchema)
def run_in_container(command: list[str], timeout_s: int = 30) -> str:
    """Run a command inside the project Docker container.

    Automatically starts the project container if it does not already exist.
    The container stays running until stop_container is called.
    """
    workspace_path = settings.WORKPLACE_FOLDER
    if workspace_path is None:
        return _build_output(
            status="error",
            exit_code=None,
            message="WORKPLACE_FOLDER must be set in the .env file.",
        )

    if not os.path.exists(workspace_path):
        return _build_output(
            status="error",
            exit_code=None,
            message=f"WORKPLACE_FOLDER path does not exist: {workspace_path}",
        )

    try:
        container_id = _get_or_start_container(workspace_path)
        return _run_command_in_container(container_id, command, timeout_s)
    except Exception as error:
        
        return _build_output(
            status="error",
            exit_code=None,
            stderr=str(error),
            message="Docker command execution failed before the command completed.",
        )

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


def _build_output(
    status: str,
    exit_code: int | None,
    stdout: str = "",
    stderr: str = "",
    message: str = "",
) -> str:
    output = DockerCommandOutputSchema(
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        message=message,
    )
    return output.model_dump_json()


def _decode_stream(stream: bytes | None) -> str:
    if stream is None:
        return ""

    return stream.decode(errors="replace")


def _format_exec_output(exit_code: int, output: tuple[bytes | None, bytes | None]) -> str:
    stdout, stderr = output
    status = "success" if exit_code == 0 else "error"
    message = "Command completed successfully."

    if exit_code == 124:
        message = "Command timed out."
    elif exit_code != 0:
        message = f"Command failed with exit code {exit_code}."

    return _build_output(
        status=status,
        exit_code=exit_code,
        stdout=_decode_stream(stdout),
        stderr=_decode_stream(stderr),
        message=message,
    )


def _start_container(
    workspace_path: str,
    container_name: str,
    image: str = DEFAULT_IMAGE,
    memory_limit: str = "256m",
    network_enabled: bool = True,
) -> str:
    client = _get_client()
    absolute_workspace_path = os.path.abspath(workspace_path)
    container: Container = client.containers.run(
        image=image,
        command=["sleep", "infinity"],
        detach=True,
        name=container_name,
        user=f"{os.getuid()}:{os.getgid()}",
        volumes={
            absolute_workspace_path: {
                "bind": DEFAULT_WORKSPACE_BIND,
                "mode": "rw",
            }
        },
        working_dir=DEFAULT_WORKSPACE_BIND,
        environment=DEFAULT_CONTAINER_ENV,
        network_disabled=not network_enabled,
        mem_limit=memory_limit,
    )

    return container.id


def _close_container(container_id: str) -> None:
    client = _get_client()
    container = client.containers.get(container_id)
    container.stop()
    container.remove()


def _run_command_in_container(
    container_id: str,
    command: Sequence[str],
    timeout_s: int,
) -> str:
    client = _get_client()
    container = client.containers.get(container_id)
    bounded_timeout_s = max(1, min(timeout_s, 300))
    timed_command = ["timeout", f"{bounded_timeout_s}s", *command]

    exit_code, output = container.exec_run(
        cmd=timed_command,
        workdir=DEFAULT_WORKSPACE_BIND,
        environment=DEFAULT_CONTAINER_ENV,
        demux=True,
    )

    return _format_exec_output(exit_code, output)

def _get_container_name() -> str:
    return getattr(settings, "AGENT_SYSTEM_CONTAINER_NAME", "agent_system_container")


def _get_or_start_container(
    workspace_path: str,
    image: str = DEFAULT_IMAGE,
    memory_limit: str = "256m",
    network_enabled: bool = True,
) -> str:
    client = _get_client()
    container_name = _get_container_name()

    try:
        container = client.containers.get(container_name)

        if container.status != "running":
            container.start()
            container.reload()

        return container.id

    except NotFound:
        return _start_container(
            workspace_path=workspace_path,
            image=image,
            memory_limit=memory_limit,
            container_name=container_name,
            network_enabled=network_enabled,
        )

    except APIError as error:
        # Handles a possible race where another process creates the container
        # between the NotFound check and container creation.
        if "Conflict" in str(error) or "already in use" in str(error):
            container = client.containers.get(container_name)

            if container.status != "running":
                container.start()
                container.reload()

            return container.id

        raise




# if __name__ == "__main__":
#     print("Testing Docker Code Execution...")

#     folder = settings.WORKPLACE_FOLDER
#     print("Initializing folder...")
#     if not os.path.exists(folder):
#         os.makedirs(folder)

#     test_file = """print("Hello, World!")
# """
#     if not os.path.exists(f"{folder}/test.py"):
#         print("Creating test file...")
#         with open(f"{folder}/test.py", "w") as f:
#             f.write(test_file)


#     while True:
#         print("Running command in container...")
#         command = input("Enter the command to run in the container (or 'exit' to quit): ").strip().split()
#         if command == ["exit"]:
#             break
#         output_json = run_in_container.invoke({"command": command, "timeout_s": 30})
#         print("Output:")
#         print(output_json)

#     # Stop the container after testing
#     print("Stopping container...")
#     print(stop_container.invoke({}))

if __name__ == "__main__":

    while True:
        print("Testing Docker Code Execution...")
        command = input("Enter the command to run in the container (or 'exit' to quit): ").strip().split()
        if command == ["exit"]:
            break
        output_json = run_in_container.invoke({"command": command, "timeout_s": 30})
        print("Output:")
        print(output_json)
    print("Exiting...")
    res = stop_container.invoke({})
    print("Container stopped:", res)

from langchain_core.tools import tool
from src.tools.tool_schemas import RunInRepositorySchema
from src.tools.docker_code_execution.tool import run_in_container, stop_container
from src.settings import settings
import os


@tool(arg_schema=RunInRepositorySchema)
def run_repository_command(command: list[str], timeout_s: int = 30) -> str:
    """Run a command inside a specific project repository that is found in the workspace.

    Automatically starts the project container if it does not already exist.
    The container stays running until stop_container is called.
    """
    if settings.REPO_NAME is None:
        return {
            "status": "error",
            "exit_code": None,
            "stderr": "REPO_NAME must be set in the .env file.",
            "message": "Repository name not configured.",
        }

    workspace_path = settings.WORKPLACE_FOLDER
    repo_path = os.path.join(workspace_path, settings.REPO_NAME)

    if not os.path.exists(repo_path):
        return {
            "status": "error",
            "exit_code": None,
            "stderr": f"Repository path does not exist in workspace: {repo_path}",
            "message": "Repository path does not exist.",
        }
    if not os.path.isdir(repo_path):
        return {
            "status": "error",
            "exit_code": None,
            "stderr": f"Repository path is not a valid directory: {repo_path}",
            "message": "Invalid repository path.",
        }
    if not os.path.exists(os.path.join(repo_path, ".git")):
        return {
            "status": "error",
            "exit_code": None,
            "stderr": f"Repository path is not a valid Git repository: {repo_path}",
            "message": "Invalid repository path.",
        }
    return run_in_container.invoke(
        {
            "command": command,
            "timeout_s": timeout_s,
            "workspace_path": repo_path,
        }
    )

    
if __name__ == "__main__":
    while True:
        user_input = input("Enter a command to run in the repository (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        command_list = user_input.split()
        result = run_repository_command.invoke({"command": command_list, "timeout_s": 30})
        # print("Command execution result:", result)
    stop_container.invoke({})
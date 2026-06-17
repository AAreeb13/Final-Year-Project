

import json
from pathlib import Path

from src.tools.file_system_tools.inspect_workplace import inspect_folder_structure
from src.tools.file_system_tools.manage_file import create_file, delete_file, edit_file

from src.tools.tool_schemas import CreateFileSchema, DeleteFileSchema, EditFileSchema, InspectFileSchema, InspectFolderStructureSchema

from src.tools.file_system_tools.inspect_file import inspect_file

from src.settings import settings
from langchain_core.tools import tool


def _resolve_file_inside_workspace_repository(relative_path: str = None, must_exist: bool = True) -> Path:
    workspace_root = Path(settings.WORKPLACE_FOLDER).expanduser().resolve()
    repository_name = settings.REPO_NAME
    if repository_name is None:
        raise ValueError("Repository name is not configured.")
    repository_path = (workspace_root / repository_name).expanduser().resolve()
    if repository_path != workspace_root and workspace_root not in repository_path.parents:
        raise ValueError("Repository path must stay inside the configured workplace folder.")
    if not repository_path.exists():
        raise ValueError("Repository folder does not exist.")
    if not repository_path.is_dir():
        raise ValueError("Repository path is not a directory.")
    if not (repository_path / ".git").exists():
        raise ValueError("Repository folder is not a valid Git repository.")

    file_path = repository_path
    if relative_path:
        file_path = (file_path / relative_path).expanduser().resolve()
    if file_path != repository_path and repository_path not in file_path.parents:
        raise ValueError("Path must stay inside the configured repository folder.")
    
    if must_exist and not file_path.exists():
        raise ValueError("File or directory does not exist.")
    
    # if file then check if it's inside the repository
    if file_path.is_file():
        if repository_name not in file_path.parts:
            raise ValueError("File must be inside the repository folder.")

    # path relative to workspace root
    resolved_path = file_path.relative_to(workspace_root)
    return resolved_path


@tool(args_schema=InspectFileSchema)
def inspect_repository_file(relative_path: str) -> str:
    """
    Inspect a file in the repository and return its contents as a JSON string.

    Input: a relative file path inside the repository.
    Output: JSON with the structure:
    {"status": "success"|"error", "content": str|None, "message": str}
    """
    try:
        file_path = _resolve_file_inside_workspace_repository(relative_path)
        print(f"Resolved file path: {file_path}")
    except ValueError as error:
        return {
            "status": "error",
            "content": None,
            "message": str(error),
        }
    print(f"Inspecting file: {file_path}")
    return inspect_file.invoke({"relative_path": str(file_path)})


@tool(args_schema=CreateFileSchema)
def create_repository_file(relative_path_to_folder: str, file_name: str) -> str:
    try:
        folder_path = _resolve_file_inside_workspace_repository(relative_path_to_folder, must_exist=False)
        print(f"Resolved folder path: {folder_path}")
    except ValueError as error:
        return {
            "status": "error",
            "message": str(error),
        }

    return create_file.invoke({"relative_path_to_folder": str(folder_path), "file_name": file_name})

@tool(args_schema=DeleteFileSchema)
def delete_repository_file(relative_path: str) -> str:

    try:
        file_path = _resolve_file_inside_workspace_repository(relative_path)
    except ValueError as error:
        return {
            "status": "error",
            "message": str(error),
        }
    
    return delete_file.invoke({"relative_path": str(file_path)})

@tool(args_schema=EditFileSchema)
def edit_repository_file(relative_path: str, new_content: str) -> str:
    print(f"Editing file at relative path: {relative_path} with new content of length {len(new_content)}")
    try:
        file_path = _resolve_file_inside_workspace_repository(relative_path)
        print(f"Resolved file path for editing: {file_path}")
    except ValueError as error:
        return {
            "status": "error",
            "message": str(error),
        }
    
    return edit_file.invoke({"relative_path": str(file_path), "new_content": new_content})

@tool(args_schema=InspectFolderStructureSchema)
def inspect_repository(relative_path: str, max_depth: int = 5, max_entries: int = 200, extra_ignored_names: list[str] = None) -> str:
    """Return a tree view of the configured Git repository or a folder,  the total number of entries and whether it's truncated 
    The output follows this format: {"tree": str, "total_entries": int, "truncated": bool}
    

    Args:
        relative_path: Optional path inside the Git repository to inspect.
        max_depth: Maximum directory depth to include.
        max_entries: Maximum number of files/directories to return.
        extra_ignored_names: Additional file or directory names to omit.
    """
    print("Inspecting repository", relative_path)
    try:
        target_path = _resolve_file_inside_workspace_repository(relative_path)
        # only repo name with relative path
        # print("target path: ", target_path)
    except ValueError as error:
        print(f"Error resolving path: {error}")
        return {
            "status": "error",
            "message": str(error),
        }
    print("invoking folder structure")
    return inspect_folder_structure.invoke({
        "relative_path": str(target_path),
        "max_depth": max_depth,
        "max_entries": max_entries,
        "extra_ignored_names": extra_ignored_names,
    })


if __name__ == "__main__":
    # inspect repo

    result = inspect_repository.invoke({"relative_path": ".", "max_depth": 3, "max_entries": 100})
    # make dicttionary from json string result and print the "tree" value
    print( result)
    print(result.get("tree"))
    # result_to_json_to_dict = json.loads(result)
    # print(result_to_json_to_dict["tree"])

    # testing the create file tool
    result = create_repository_file.invoke({"relative_path_to_folder": "", "file_name": "example.txt"})
    print(result)

    # testing the edit file tool
    result = edit_repository_file.invoke({"relative_path": "example.txt", "new_content": "This is an example file."})
    print(result)

    # testing the inspect file tool
    result = inspect_repository_file.invoke({"relative_path": "example.txt"})
    print(result)

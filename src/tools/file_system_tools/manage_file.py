from datetime import date
import json
from pathlib import Path

from langchain_core.tools import tool
from src.tools.tool_schemas import CreateFileSchema, DeleteFileSchema, EditFileSchema

from src.settings import settings


@tool(args_schema=CreateFileSchema)
def create_file(relative_path_to_folder: str, file_name: str) -> str:
    """
    Create a file in the workplace folder with the specified content.

    Input: 
        relative_path_to_folder: str - a relative path to the folder where the file should be created.
        file_name: str - the name of the file to create.
    Output: JSON with the structure:
    {"status": "success"|"error", "message": str}
    """
    workspace_root = Path(settings.WORKPLACE_FOLDER).expanduser().resolve()
    file_path = (workspace_root / relative_path_to_folder / file_name).expanduser().resolve()

    if file_path.exists():
        return json.dumps({"status": "error", "message": f"File already exists: {file_name}"})

    if file_path != workspace_root and workspace_root not in file_path.parents:
        return json.dumps({"status": "error", "message": "Path must stay inside the configured workplace folder."})

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch(exist_ok=False)
    except Exception as error:
        return json.dumps({"status": "error", "message": f"Failed to create file: {str(error)}"})

    return json.dumps({"status": "success", "message": "File created successfully. check the repository structure to see if it is in the right place."})

# will not give this to the agent for now as it can be dangerous, but it can be useful for testing and debugging
@tool(args_schema=DeleteFileSchema)
def delete_file(relative_path: str) -> str:
    """
    Delete a file in the workplace folder.

    Input: 
        relative_path: str - a relative file path inside the workplace folder.
    Output: JSON with the structure:
    {"status": "success"|"error", "message": str}
    """
    workspace_root = Path(settings.WORKPLACE_FOLDER).expanduser().resolve()
    file_path = (workspace_root / relative_path).expanduser().resolve()

    if not file_path.exists():
        return json.dumps({"status": "error", "message": f"File not found: {relative_path}"})

    if not file_path.is_file():
        return json.dumps({"status": "error", "message": f"Path is not a file: {relative_path}"})

    if file_path != workspace_root and workspace_root not in file_path.parents:
        return json.dumps({"status": "error", "message": "Path must stay inside the configured workplace folder."})

    try:
        file_path.unlink()
    except Exception as error:
        return json.dumps({"status": "error", "message": f"Failed to delete file: {str(error)}"})

    return json.dumps({"status": "success", "message": "File deleted successfully."})

@tool(args_schema=EditFileSchema)
def edit_file(relative_path: str, new_content: str) -> str:
    """
    Edit a file in the workplace folder by replacing its contents with new content.
    The file must already exist.

    Input: 
        relative_path: str - a relative file path inside the workplace folder.
        new_content: str - the new content to write to the file.

    Output: JSON with the structure:
    {"status": "success"|"error", "message": str}
    """
    workspace_root = Path(settings.WORKPLACE_FOLDER).expanduser().resolve()
    file_path = (workspace_root / relative_path).expanduser().resolve()

    if file_path != workspace_root and workspace_root not in file_path.parents:
        return json.dumps({"status": "error", "message": "Path must stay inside the configured workplace folder."})

    if not file_path.exists():
        return json.dumps({"status": "error", "message": f"File not found: {relative_path}"})
    
    if not file_path.is_file():
        return json.dumps({"status": "error", "message": f"Path is not a file: {relative_path}"})

    try:
        file_path.write_text(new_content, encoding="utf-8")
    
    except Exception as error:
        return json.dumps({"status": "error", "message": f"Failed to write to file: {str(error)}"})

    return json.dumps({"status": "success", "message": "File edited successfully."})

if __name__ == "__main__":
    # Example usage

    #testing the create file tool
    result = create_file.invoke({"relative_path_to_folder": "repository", "file_name": "example.txt"})
    print(result)

    # testing the delete file tool
    result = delete_file.invoke({"relative_path": "repository/example.txt"})
    print(result)

    # testing the create file tool again to create the file for the edit test
    result = create_file.invoke({"relative_path_to_folder": "repository", "file_name": "example.txt"})
    print(result)
    # testing the edit file tool
    result = edit_file.invoke(
        {"relative_path": "repository/example.txt",
          "new_content": f"# Agent: {date.today()} \nThis is the new new content of the file."})
    print(result)





# creating folders and deleting folders

from pathlib import Path

from langchain_core.tools import tool
from tools.tool_schemas import CreateFolderSchema, DeleteFolderSchema
from src.settings import settings

@tool(args_schema=CreateFolderSchema)
def create_folder(relative_path: str) -> str:
    """
    Create a folder in the workplace.

    Input:
        relative_path: str - a relative folder path inside the workplace folder.
    
    Output: JSON with the structure:
    {"status": "success"|"error", "message": str}
    """

    # see if the folder already exists
    workspace_root = Path(settings.WORKPLACE_FOLDER).expanduser().resolve()
    folder_path = (workspace_root / relative_path).expanduser().resolve()

    if folder_path.exists():
        return {"status": "error", "message": f"Folder already exists: {relative_path}"}
    if folder_path != workspace_root and workspace_root not in folder_path.parents:
        return {"status": "error", "message": "Path must stay inside the configured workplace folder."}
    try:
        folder_path.mkdir(parents=True, exist_ok=False)
    except Exception as error:
        return {"status": "error", "message": f"Failed to create folder: {str(error)}"}
    return {"status": "success", "message": "Folder created successfully."}

@tool(args_schema=DeleteFolderSchema)
def delete_folder(relative_path: str) -> str:
    """
    Delete a folder in the workplace.

    Input:
        relative_path: str - a relative folder path inside the workplace folder.
    
    Output: JSON with the structure:
    {"status": "success"|"error", "message": str}
    """

    result, folder_path = _validate_folder_delete(relative_path)
    if result["status"] == "error":
        return result

    try:
        folder_path.rmdir()
    except Exception as error:
        return {"status": "error", "message": f"Failed to delete folder: {str(error)}"}

    return {"status": "success", "message": "Folder deleted successfully."}


def _validate_folder_delete(relative_path: str):
    workspace_root = Path(settings.WORKPLACE_FOLDER).expanduser().resolve()
    folder_path = (workspace_root / relative_path).expanduser().resolve()

    if not folder_path.exists():
        return {"status": "error", "message": f"Folder does not exist: {relative_path}"}

    if folder_path == workspace_root:
        return {"status": "error", "message": "Cannot delete the workplace folder."}

    if folder_path != workspace_root and workspace_root not in folder_path.parents:
        return {"status": "error", "message": "Path must stay inside the configured workplace folder."}
    if not folder_path.is_dir():
        return {"status": "error", "message": f"Path is not a folder: {relative_path}"}
    
    return {"status": "success", "message": "Folder is valid."} , folder_path



if __name__ == "__main__":
    
    print(create_folder.invoke({"relative_path": "test_folder"}))
    # sleep for a second to make sure the folder is created before trying to delete it
    import time
    time.sleep(10)
    print(delete_folder.invoke({"relative_path": "test_folder"}))
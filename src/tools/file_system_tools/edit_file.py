import json
from pathlib import Path

from langchain_core.tools import tool

from src.settings import settings


@tool
def edit_file(relative_path: str, new_content: str, new_file: bool = False) -> str:
    """
    Edit a file in the workplace folder by replacing its contents with new content.

    Input: 
        relative_path: str - a relative file path inside the workplace folder.
        new_content: str - the new content to write to the file.
        new_file: bool (optional, default False) - if True, allows creating a new file if it doesn't exist.
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
        if not new_file:
            return json.dumps({"status": "error", "message": f"Path is not a file: {relative_path}"})
        
        # create a new file
        file_path.touch()

    try:
        file_path.write_text(new_content, encoding="utf-8")
    
    except Exception as error:
        return json.dumps({"status": "error", "message": f"Failed to write to file: {str(error)}"})

    return json.dumps({"status": "success", "message": "File edited successfully."})
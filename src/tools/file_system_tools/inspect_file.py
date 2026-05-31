import json
from pathlib import Path

from langchain_core.tools import tool
from tools.tool_schemas import InspectFileSchema

from src.settings import settings


def _json_response(status: str, content: str | None, message: str) -> str:
    return json.dumps(
        {
            "status": status,
            "content": content,
            "message": message,
        }
    )


def _resolve_file_inside_workspace(relative_path: str) -> Path:
    workspace_root = Path(settings.WORKPLACE_FOLDER).expanduser().resolve()
    file_path = (workspace_root / relative_path).expanduser().resolve()

    if file_path != workspace_root and workspace_root not in file_path.parents:
        raise ValueError("Path must stay inside the configured workplace folder.")

    return file_path


@tool(args_schema=InspectFileSchema)
def inspect_file(relative_path: str) -> str:
    """
    Inspect a file in the workplace folder and return its contents as a JSON string.

    Input: a relative file path inside the workplace folder.
    Output: JSON with the structure:
    {"status": "success"|"error", "content": str|None, "message": str}
    """
    try:
        file_path = _resolve_file_inside_workspace(relative_path)
    except ValueError as error:
        return _json_response("error", None, str(error))

    if not file_path.exists():
        return _json_response("error", None, f"File not found: {relative_path}")

    if not file_path.is_file():
        return _json_response("error", None, f"Path is not a file: {relative_path}")

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return _json_response("error", None, f"File is not valid UTF-8 text: {relative_path}")

    return _json_response("success", content, "File read successfully.")


if __name__ == "__main__":
    print(inspect_file.invoke({"relative_path": "test/idk.txt"}))

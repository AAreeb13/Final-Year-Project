from pydantic import BaseModel


class RunInContainerSchema(BaseModel):
    """
    Schema for the run_in_container tool.
    """
    command: list[str]
    timeout_s: int = 30

class InspectFolderStructureSchema(BaseModel):
    """
    Schema for the inspect_folder_structure tool.
    """
    relative_path: str | None = None
    max_depth:  int = 4
    max_entries: int = 200
    extra_ignored_names: list[str] | None = None

class EditFileSchema(BaseModel):
    """
    Schema for the edit_file tool.
    """
    relative_path: str
    new_content: str

class CreateFileSchema(BaseModel):
    """
    Schema for the create_file tool.
    """
    relative_path_to_folder: str
    file_name: str

class DeleteFileSchema(BaseModel):
    """
    Schema for the delete_file tool.
    """
    relative_path: str

class InspectFileSchema(BaseModel):
    """
    Schema for the inspect_file tool.
    """
    relative_path: str

class CreateFolderSchema(BaseModel):
    """
    Schema for the create_folder tool.
    """
    relative_path: str


class DeleteFolderSchema(BaseModel):
    """
    Schema for the delete_folder tool.
    """
    relative_path: str

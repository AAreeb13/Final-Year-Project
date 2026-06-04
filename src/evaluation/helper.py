from pathlib import Path


def load_yaml_file(file_path: Path) -> dict:
    import yaml

    if not file_path.exists():
        raise ValueError(f"YAML file not found at path: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"Provided path is not a file: {file_path}")
    
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file at {file_path}: {str(e)}") from e
    except Exception as e:
        raise ValueError(f"Unexpected error loading YAML file at {file_path}: {str(e)}") from e

from pathlib import Path
from typing import Any

from src.evaluation.sdlc_eval_schema import (
    ComponentSpec,
    GraphDependencySpec,
    HLArchitectureSpec,
    ImplementationStep,
    ModuleSpec,
    ProjectInfoSpec,
    ProjectSpec,
    RepositoryStructure,
    RequirementsSpec,
    TestPlan,
)


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

class FileNames(dict):
    PROJECT_INFO = "project_info.yaml"
    DESIGN = "design.yaml"
    REQUIREMENTS = "requirements.yaml"
    IMPLEMENTATION_PLAN = "implementation_plan.yaml"
    REPOSITORY_STRUCTURE = "repository_structure.yaml"
    GRAPH_DEPENDENCY_SPEC = "graph_dependency_spec.yaml"
    COMPONENTS = "components.yaml"
    MODULES = "modules.yaml"
    TEST_PLAN = "test_plan.yaml"

    def __init__(self):
        for key, value in self.__class__.__dict__.items():
            if key.isupper():
                self[key] = value


def collate_project_yaml(
    project_id: str,
    dataset_dir: str | Path | None = None,
    *,
    warn_missing: bool = False,
) -> dict[str, Any]:
    """
    Loads all YAML files for a dataset project folder.

    Returns raw loaded data keyed by FileNames keys.
    """
    if dataset_dir is None:
        from src.settings import settings

        if settings.DATASET_DIRECTORY is None:
            raise ValueError("DATASET_DIRECTORY must be set in the .env file")
        dataset_dir = settings.DATASET_DIRECTORY

    file_names = FileNames()
    dataset_root = Path(dataset_dir).expanduser().resolve()
    project_dir = dataset_root / "dataset" / project_id

    if not project_dir.exists():
        raise ValueError(f"Project directory does not exist: {project_dir}")

    project_data: dict[str, Any] = {}

    for name, filename in file_names.items():
        file_path = project_dir / filename

        try:
            project_data[name] = load_yaml_file(file_path)
        except ValueError as e:
            if warn_missing:
                print(f"Warning: {str(e)}. Skipping file: {file_path}")
            project_data[name] = None

    return project_data


def build_project_spec_from_yaml(
    project_id: str,
    dataset_dir: str | Path | None = None,
    *,
    debug: bool = False,
    warn_missing: bool = False,
) -> ProjectSpec:
    """
    Converts separate YAML files inside one dataset project folder into
    a single validated ProjectSpec.
    """
    raw = collate_project_yaml(
        project_id,
        dataset_dir=dataset_dir,
        warn_missing=warn_missing,
    )

    project_info_data = _unwrap(
        raw.get("PROJECT_INFO"),
        possible_keys=["ProjectInfoSpec", "project_info", "Project"],
        default={},
    )

    project_info = ProjectInfoSpec.model_validate(project_info_data)

    requirements_data = _unwrap(
        raw.get("REQUIREMENTS"),
        possible_keys=["RequirementsSpec", "requirements"],
        default={},
    )

    design_data = _unwrap(
        raw.get("DESIGN"),
        possible_keys=["HLArchitectureSpec", "ArchitectureSpec", "high_level_design", "design"],
        default={},
    )

    components_data = _unwrap(
        raw.get("COMPONENTS"),
        possible_keys=["components", "ComponentSpec"],
        default=[],
    )

    modules_data = _unwrap(
        raw.get("MODULES"),
        possible_keys=["modules", "ModuleSpec"],
        default=[],
    )

    graph_data = _unwrap(
        raw.get("GRAPH_DEPENDENCY_SPEC"),
        possible_keys=["GraphDependencySpec", "module_dependency_graph", "DependencyGraph"],
        default={},
    )

    repository_structure_data = _unwrap(
        raw.get("REPOSITORY_STRUCTURE"),
        possible_keys=["RepositoryStructure", "repository_structure"],
        default={},
    )

    implementation_plan_data = _unwrap(
        raw.get("IMPLEMENTATION_PLAN"),
        possible_keys=["implementation_plan", "ImplementationStep"],
        default=[],
    )

    test_plan_data = _unwrap(
        raw.get("TEST_PLAN"),
        possible_keys=["TestPlan", "test_plan"],
        default={},
    )

    project_spec = ProjectSpec(
        project_id=project_info.project_id or project_id,
        project_prompt=project_info.project_prompt,
        project_type=project_info.project_type,
        difficulty=project_info.difficulty,

        requirements=RequirementsSpec.model_validate(requirements_data),
        high_level_design=HLArchitectureSpec.model_validate(
            _normalise_design_keys(design_data)
        ),

        components=_validate_list(components_data, ComponentSpec),
        modules=_validate_modules(modules_data, debug=debug),
        module_dependency_graph=GraphDependencySpec.model_validate(
            _normalise_graph_dependency_spec(graph_data)
        ),

        repository_structure=RepositoryStructure.model_validate(
            _normalise_repository_structure(repository_structure_data)
        ),
        implementation_plan=_validate_list(
            _normalise_implementation_plan(implementation_plan_data),
            ImplementationStep,
        ),
        test_plan=TestPlan.model_validate(test_plan_data),
    )
    return project_spec


def _unwrap(
    data: Any,
    possible_keys: list[str],
    default: Any,
) -> Any:
    """
    Handles YAML files that may either be:

    1. direct:
       functional:
         - ...

    2. wrapped:
       RequirementsSpec:
         functional:
           - ...

    3. named:
       requirements:
         functional:
           - ...
    """
    if data is None:
        return default

    if not isinstance(data, dict):
        return data

    for key in possible_keys:
        if key in data:
            return data[key]

    return data


def _validate_list(data: Any, model_cls: type) -> list:
    """
    Validates either:

    - a list of objects
    - a single object
    - an empty/missing value
    """
    if data is None:
        return []

    if isinstance(data, dict):
        return [model_cls.model_validate(data)]

    if isinstance(data, list):
        return [model_cls.model_validate(item) for item in data]

    raise ValueError(f"Expected list or dict for {model_cls.__name__}, got {type(data)}")

def _normalise_design_keys(data: dict[str, Any]) -> dict[str, Any]:
    """
    Normalises design YAML into the HLArchitectureSpec schema.

    Handles:
    - Relationships -> relationships
    - Technologies -> technologies
    - responsibility -> responsibilities
    - responsibility: str -> responsibilities: list[str]
    """
    if not isinstance(data, dict):
        return {}

    normalised = dict(data)

    if "Relationships" in normalised and "relationships" not in normalised:
        normalised["relationships"] = normalised.pop("Relationships")
    if "Technologies" in normalised and "technologies" not in normalised:
        normalised["technologies"] = normalised.pop("Technologies")
    components = normalised.get("components", [])

    if isinstance(components, list):
        normalised_components = []
        for component in components:
            if not isinstance(component, dict):
                continue
            component = dict(component)

            if "responsibility" in component and "responsibilities" not in component:
                responsibility = component.pop("responsibility")
                if isinstance(responsibility, list):
                    component["responsibilities"] = responsibility
                else:
                    component["responsibilities"] = [str(responsibility)]
            normalised_components.append(component)
        normalised["components"] = normalised_components
    return normalised


def _normalise_graph_dependency_spec(data: Any) -> dict[str, Any]:
    """
    Normalises graph dependency YAML into GraphDependencySpec.

    Handles:
    - graph_dependencies: {nodes: [...], edges: [...]}
    - graph_dependencies: [{Relationships: [{source, target, ...}]}]
    - Relationships/relationships lists directly at the top level
    """
    if data is None:
        return {}

    if not isinstance(data, dict):
        return {}

    graph_data = data.get("graph_dependencies", data)

    if isinstance(graph_data, dict):
        if "nodes" in graph_data or "edges" in graph_data:
            return {
                "nodes": _normalise_str_list(graph_data.get("nodes", [])),
                "edges": _normalise_graph_edges(graph_data.get("edges", [])),
            }

        relationships = graph_data.get("Relationships", graph_data.get("relationships"))
        if relationships is not None:
            edges = _normalise_graph_edges(relationships)
            return {
                "nodes": _nodes_from_edges(edges),
                "edges": edges,
            }

    if isinstance(graph_data, list):
        edges = []

        for item in graph_data:
            if not isinstance(item, dict):
                continue
            relationships = item.get("Relationships", item.get("relationships"))
            if relationships is not None:
                edges.extend(_normalise_graph_edges(relationships))
            elif "source" in item and "target" in item:
                edges.extend(_normalise_graph_edges([item]))

        return {
            "nodes": _nodes_from_edges(edges),
            "edges": edges,
        }

    return {}


def _normalise_graph_edges(edges_data: Any) -> list[dict[str, str]]:
    if edges_data is None:
        return []

    if isinstance(edges_data, dict):
        edges_data = [edges_data]

    if not isinstance(edges_data, list):
        return []

    edges = []

    for edge in edges_data:
        if not isinstance(edge, dict):
            continue
        edges.append(
            {
                "source": str(edge.get("source", "")),
                "target": str(edge.get("target", "")),
            }
        )

    return edges


def _nodes_from_edges(edges: list[dict[str, str]]) -> list[str]:
    nodes = []

    for edge in edges:
        for node in (edge.get("source", ""), edge.get("target", "")):
            if node and node not in nodes:
                nodes.append(node)

    return nodes


def _normalise_str_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    return [str(value)]


def _normalise_repository_structure(data: Any) -> dict[str, Any]:
    """
    Normalises repository structure YAML into RepositoryStructure.

    Handles:
    - schema-native {"directories": [...]}
    - wrapped {"repository_structure": ...}
    - simple path lists such as ["src/", "src/app.py", "tests/"]
    """
    if data is None:
        return {}

    if isinstance(data, dict):
        if "directories" in data:
            return data
        if "repository_structure" in data:
            return _normalise_repository_structure(data["repository_structure"])
        return data

    if not isinstance(data, list):
        return {}

    directories: dict[str, dict[str, Any]] = {}

    def ensure_directory(path: str) -> dict[str, Any]:
        clean_path = path.strip("/")
        name = clean_path.rsplit("/", 1)[-1] if clean_path else "."
        parent = clean_path.rsplit("/", 1)[0] if "/" in clean_path else None

        directory = directories.setdefault(
            clean_path,
            {
                "name": name,
                "parent": parent,
                "children": [],
                "files": [],
            },
        )

        if parent is not None:
            parent_directory = ensure_directory(parent)
            if name not in parent_directory["children"]:
                parent_directory["children"].append(name)

        return directory

    for raw_path in data:
        if not isinstance(raw_path, str):
            continue

        path = raw_path.strip()
        if not path:
            continue

        if path.endswith("/"):
            ensure_directory(path)
            continue

        parent_path, file_name = (
            path.rsplit("/", 1) if "/" in path else ("", path)
        )
        parent_directory = ensure_directory(parent_path)

        if not any(file_item["name"] == file_name for file_item in parent_directory["files"]):
            parent_directory["files"].append(
                {
                    "name": file_name,
                    "modules": [],
                }
            )

    return {"directories": list(directories.values())}


def _normalise_implementation_plan(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []

    if isinstance(data, dict):
        if "implementation_plan" in data:
            return _normalise_implementation_plan(data["implementation_plan"])
        data = [data]

    if not isinstance(data, list):
        return []

    allowed_fields = set(ImplementationStep.model_fields)
    normalised_steps = []

    for item in data:
        if isinstance(item, str):
            normalised_steps.append(
                {
                    "step_id": "",
                    "action": item,
                }
            )
            continue

        if not isinstance(item, dict):
            continue

        step = dict(item)

        if "step_id" in step and step["step_id"] is not None:
            step["step_id"] = str(step["step_id"])

        if "description" in step and "action" not in step:
            step["action"] = str(step.pop("description"))
        else:
            step.pop("description", None)

        normalised_steps.append(
            {
                key: value
                for key, value in step.items()
                if key in allowed_fields
            }
        )

    return normalised_steps

def _debug_print(title: str, data: Any) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    try:
        import json
        print(json.dumps(data, indent=2, default=str))
    except Exception:
        print(data)

def _validate_modules(data: Any, *, debug: bool = False) -> list[ModuleSpec]:
    """
    Normalises and validates module YAML.

    Handles:
    - signatures: dict -> signatures: [dict]
    - removes unsupported module fields such as technologies
    - optionally prints useful debugging information
    """
    if data is None:
        return []

    if isinstance(data, dict):
        if "modules" in data:
            data = data["modules"]
        elif "ModuleSpec" in data:
            data = data["ModuleSpec"]
        else:
            data = [data]

    if not isinstance(data, list):
        raise ValueError(f"Expected modules to be list or dict, got {type(data)}")

    validated_modules: list[ModuleSpec] = []

    allowed_module_fields = set(ModuleSpec.model_fields)

    for index, raw_module in enumerate(data):
        if debug:
            _debug_print(f"Raw module before normalisation [{index}]", raw_module)

        if not isinstance(raw_module, dict):
            raise ValueError(
                f"Module at index {index} must be a dict, got {type(raw_module)}"
            )

        module = dict(raw_module)

        # Remove unsupported fields from ModuleSpec.
        removed_fields = {}

        for key in list(module.keys()):
            if key not in allowed_module_fields:
                removed_fields[key] = module.pop(key)

        if removed_fields:
            if debug:
                _debug_print(
                    f"Removed unsupported fields from module [{index}]",
                    removed_fields,
                )

        # Convert signatures from a single dict into a list.
        if "signatures" in module and isinstance(module["signatures"], dict):
            if debug:
                print(
                    f"Normalising module [{index}] signatures: "
                    "dict -> list[dict]"
                )
            module["signatures"] = [module["signatures"]]

        # Ensure missing signatures becomes empty list.
        if "signatures" not in module or module["signatures"] is None:
            module["signatures"] = []
        # Normalise each signature.
        module["signatures"] = [
            _normalise_signature(signature, module_index=index, signature_index=sig_index)
            for sig_index, signature in enumerate(module["signatures"])
        ]

        if debug:
            _debug_print(f"Module after normalisation [{index}]", module)

        try:
            validated_module = ModuleSpec.model_validate(module)
            validated_modules.append(validated_module)
        except Exception as exc:
            if debug:
                _debug_print(f"FAILED module validation [{index}]", module)
            print(f"Validation error for module [{index}]: {exc}")
            raise

    return validated_modules

def _normalise_signature(
    signature: Any,
    module_index: int,
    signature_index: int,
) -> dict[str, Any]:
    if not isinstance(signature, dict):
        raise ValueError(
            f"Signature {signature_index} in module {module_index} "
            f"must be a dict, got {type(signature)}"
        )

    signature = dict(signature)

    # Some YAML may use parameter/params instead of inputs.
    if "parameters" in signature and "inputs" not in signature:
        signature["inputs"] = signature.pop("parameters")

    if "params" in signature and "inputs" not in signature:
        signature["inputs"] = signature.pop("params")

    # SignatureSpec has a singular string `output`, while generated YAML often
    # uses `outputs` or `returns` with dict/list shapes.
    if "output" not in signature or signature["output"] is None:
        for output_alias in ("outputs", "returns", "return"):
            if output_alias in signature:
                signature["output"] = _normalise_signature_output(
                    signature.pop(output_alias)
                )
                break

    if "inputs" not in signature or signature["inputs"] is None:
        signature["inputs"] = []

    # If inputs is a dict, convert it to a list of ParamSpec-like objects.
    if isinstance(signature["inputs"], dict):
        converted_inputs = []

        for param_name, param_type in signature["inputs"].items():
            converted_inputs.append(
                {
                    "name": str(param_name),
                    "type": str(param_type),
                    "required": True,
                }
            )

        signature["inputs"] = converted_inputs

    if isinstance(signature["inputs"], list):
        signature["inputs"] = [
            _normalise_signature_input(
                input_value,
                module_index=module_index,
                signature_index=signature_index,
                input_index=input_index,
            )
            for input_index, input_value in enumerate(signature["inputs"])
        ]

    return signature


def _normalise_signature_input(
    input_value: Any,
    module_index: int,
    signature_index: int,
    input_index: int,
) -> dict[str, Any]:
    if isinstance(input_value, str):
        return {
            "name": input_value,
            "type": "",
            "required": True,
        }

    if isinstance(input_value, dict):
        if "name" in input_value:
            return dict(input_value)

        if len(input_value) == 1:
            name, value_type = next(iter(input_value.items()))
            return {
                "name": str(name),
                "type": str(value_type),
                "required": True,
            }

    raise ValueError(
        f"Input {input_index} in signature {signature_index} of module "
        f"{module_index} must be a string, ParamSpec dict, or single-key dict; "
        f"got {type(input_value)}"
    )


def _normalise_signature_output(output_value: Any) -> str:
    if output_value is None:
        return ""

    if isinstance(output_value, str):
        return output_value

    if isinstance(output_value, dict):
        return ", ".join(
            f"{name}: {value_type}"
            for name, value_type in output_value.items()
        )

    if isinstance(output_value, list):
        return ", ".join(_normalise_signature_output(item) for item in output_value)

    return str(output_value)
if __name__ == "__main__":
    project_id = "snake_game"
    project_data = build_project_spec_from_yaml(project_id)
    print(project_data.model_dump_json(indent=2))

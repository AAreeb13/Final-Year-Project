from __future__ import annotations

from typing import Optional

from pydantic import Field, model_validator

from src.evaluation.sdlc_eval_schema import (
    ComponentSpec,
    DirectorySpec,
    FileSpec,
    GraphDependencySpec,
    HLComponentSpec,
    HLArchitectureSpec,
    ImplementationStep,
    ModuleSpec,
    RelationshipSpec,
    RepositoryStructure,
    RequirementsSpec,
    SignatureSpec,
    StrictBaseModel,
    TestPlan,
)


class GeneratedFile(StrictBaseModel):
    path: str = ""
    content: str = ""
    purpose: Optional[str] = None


class ExecutionResult(StrictBaseModel):
    command: str = ""
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    summary: str = ""


class ToolCallRecord(StrictBaseModel):
    tool_name: str = ""
    args_summary: str = ""
    success: bool = False
    result_summary: str = ""
    error: Optional[str] = None


class SystemRunMetrics(StrictBaseModel):
    total_steps: int = 0
    agent_messages: int = 0
    tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    execution_commands: int = 0
    failed_execution_commands: int = 0
    repair_attempts: int = 0
    runtime_seconds: Optional[float] = None


class SystemRunOutput(StrictBaseModel):
    project_id: Optional[str] = None
    system_name: str = ""
    status: str = "unknown"

    requirements: RequirementsSpec = Field(default_factory=RequirementsSpec)
    high_level_design: HLArchitectureSpec = Field(default_factory=HLArchitectureSpec)

    components: list[ComponentSpec] = Field(default_factory=list)
    modules: list[ModuleSpec] = Field(default_factory=list)
    module_dependency_graph: GraphDependencySpec = Field(default_factory=GraphDependencySpec)

    repository_structure: RepositoryStructure = Field(default_factory=RepositoryStructure)
    implementation_plan: list[ImplementationStep] = Field(default_factory=list)
    test_plan: TestPlan = Field(default_factory=TestPlan)

    generated_files: list[GeneratedFile] = Field(default_factory=list)
    execution_results: list[ExecutionResult] = Field(default_factory=list)

    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    metrics: SystemRunMetrics = Field(default_factory=SystemRunMetrics)

    summary: str = ""
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalise_common_agent_shapes(cls, data):
        if not isinstance(data, dict):
            return data

        normalised = dict(data)

        def normalise_str_list(value):
            if value is None:
                return []
            if isinstance(value, str):
                return [
                    item.strip()
                    for item in value.split(",")
                    if item.strip()
                ]
            if isinstance(value, list):
                return [
                    str(item)
                    for item in value
                    if item is not None and not isinstance(item, dict)
                ] + [
                    item
                    for item in value
                    if isinstance(item, dict)
                ]
            return value

        def drop_extra_fields(value, model_cls):
            if not isinstance(value, dict):
                return value

            allowed_fields = set(model_cls.model_fields)
            return {
                key: item
                for key, item in value.items()
                if key in allowed_fields
            }

        def normalise_named_item(value, model_cls, fallback_field="name"):
            if isinstance(value, str):
                return {fallback_field: value}
            if isinstance(value, dict):
                return drop_extra_fields(value, model_cls)
            return value

        def normalise_items(value, model_cls, fallback_field="name"):
            if value is None:
                return []
            if isinstance(value, dict):
                return [
                    normalise_named_item(item, model_cls, fallback_field)
                    for item in value.values()
                ]
            if isinstance(value, list):
                return [
                    normalise_named_item(item, model_cls, fallback_field)
                    for item in value
                ]
            return [normalise_named_item(value, model_cls, fallback_field)]

        def normalise_signature_inputs(signatures):
            if not isinstance(signatures, list):
                return signatures

            normalised_signatures = []
            for signature in signatures:
                if not isinstance(signature, dict):
                    normalised_signatures.append(signature)
                    continue

                normalised_signature = dict(signature)
                inputs = normalised_signature.get("inputs")
                if isinstance(inputs, list):
                    normalised_signature["inputs"] = [
                        {"name": input_value, "type": ""}
                        if isinstance(input_value, str)
                        else input_value
                        for input_value in inputs
                    ]
                normalised_signature = drop_extra_fields(
                    normalised_signature,
                    SignatureSpec,
                )
                normalised_signatures.append(normalised_signature)

            return normalised_signatures

        def normalise_module(module):
            if isinstance(module, str):
                return {"name": module}
            if not isinstance(module, dict):
                return module

            normalised_module = dict(module)
            if "responsibilities" in normalised_module:
                normalised_module["responsibilities"] = normalise_str_list(
                    normalised_module["responsibilities"]
                )
            if "dependencies" in normalised_module:
                normalised_module["dependencies"] = normalise_str_list(
                    normalised_module["dependencies"]
                )
            if "signatures" in normalised_module:
                normalised_module["signatures"] = normalise_signature_inputs(
                    normalised_module["signatures"]
                )
            return drop_extra_fields(normalised_module, ModuleSpec)

        def normalise_component(component):
            if isinstance(component, str):
                return {"name": component}
            if not isinstance(component, dict):
                return component

            normalised_component = dict(component)
            for field_name in (
                "responsibilities",
                "inputs",
                "outputs",
                "dependencies",
                "technologies",
            ):
                if field_name in normalised_component:
                    normalised_component[field_name] = normalise_str_list(
                        normalised_component[field_name]
                    )
            return drop_extra_fields(normalised_component, ComponentSpec)

        def normalise_high_level_design(value):
            if not isinstance(value, dict):
                return value

            normalised_design = dict(value)
            if "components" in normalised_design:
                normalised_design["components"] = normalise_items(
                    normalised_design["components"],
                    HLComponentSpec,
                )
            if "relationships" in normalised_design:
                normalised_design["relationships"] = normalise_items(
                    normalised_design["relationships"],
                    RelationshipSpec,
                    fallback_field="relationship_type",
                )
            if "technologies" in normalised_design:
                normalised_design["technologies"] = normalise_str_list(
                    normalised_design["technologies"]
                )
            return drop_extra_fields(normalised_design, HLArchitectureSpec)

        def normalise_repository_structure(value):
            if not isinstance(value, dict):
                return value

            normalised_structure = dict(value)
            directories = []
            for directory in normalise_items(
                normalised_structure.get("directories"),
                DirectorySpec,
            ):
                if not isinstance(directory, dict):
                    directories.append(directory)
                    continue

                normalised_directory = dict(directory)
                if "children" in normalised_directory:
                    normalised_directory["children"] = normalise_str_list(
                        normalised_directory["children"]
                    )
                if "files" in normalised_directory:
                    files = []
                    for file_item in normalise_items(
                        normalised_directory["files"],
                        FileSpec,
                    ):
                        if isinstance(file_item, dict) and "modules" in file_item:
                            file_item["modules"] = normalise_str_list(
                                file_item["modules"]
                            )
                        files.append(file_item)
                    normalised_directory["files"] = files
                directories.append(drop_extra_fields(normalised_directory, DirectorySpec))
            normalised_structure["directories"] = directories
            return drop_extra_fields(normalised_structure, RepositoryStructure)

        def normalise_test_plan(value):
            if not isinstance(value, dict):
                return value

            normalised_plan = dict(value)
            for field_name in ("unit_tests", "integration_tests", "commands"):
                if field_name in normalised_plan:
                    normalised_plan[field_name] = normalise_str_list(
                        normalised_plan[field_name]
                    )
            return drop_extra_fields(normalised_plan, TestPlan)

        requirements = dict(normalised.get("requirements") or {})
        if "functional_requirements" in normalised:
            requirements.setdefault(
                "functional",
                normalised.pop("functional_requirements") or [],
            )
        if "non_functional_requirements" in normalised:
            requirements.setdefault(
                "non_functional",
                normalised.pop("non_functional_requirements") or [],
            )
        if requirements:
            for field_name in (
                "functional",
                "non_functional",
                "constraints",
                "assumptions",
                "out_of_scope",
            ):
                if field_name in requirements:
                    requirements[field_name] = normalise_str_list(
                        requirements[field_name]
                    )
            requirements = drop_extra_fields(requirements, RequirementsSpec)
            normalised["requirements"] = requirements

        high_level_design = dict(normalised.get("high_level_design") or {})
        if "relationships" in normalised:
            high_level_design.setdefault(
                "relationships",
                normalised.pop("relationships") or [],
            )
        if high_level_design:
            normalised["high_level_design"] = normalise_high_level_design(
                high_level_design
            )

        summary_parts = []
        existing_summary = normalised.get("summary")
        if existing_summary:
            summary_parts.append(str(existing_summary))
        for legacy_field in (
            "system_design",
            "solid_principles_used",
            "solid_principles_jeopardised",
        ):
            legacy_value = normalised.pop(legacy_field, None)
            if legacy_value:
                summary_parts.append(f"{legacy_field}: {legacy_value}")
        if summary_parts:
            normalised["summary"] = "\n".join(summary_parts)

        component_fields = set(ComponentSpec.model_fields)
        module_fields = set(ModuleSpec.model_fields)
        modules = [
            normalise_module(module)
            for module in normalised.get("modules") or []
        ]
        cleaned_components = []

        for component in normalised.get("components") or []:
            if not isinstance(component, dict):
                cleaned_components.append(normalise_component(component))
                continue

            if "component" in component or "signatures" in component:
                modules.append(
                    normalise_module(
                        {
                            key: value
                            for key, value in component.items()
                            if key in module_fields
                        }
                    )
                )

            cleaned_component = normalise_component(
                {
                    key: value
                    for key, value in component.items()
                    if key in component_fields
                }
            )
            if cleaned_component:
                cleaned_components.append(cleaned_component)

        normalised["components"] = cleaned_components
        normalised["modules"] = modules

        if "module_dependency_graph" in normalised and isinstance(
            normalised["module_dependency_graph"],
            dict,
        ):
            graph = dict(normalised["module_dependency_graph"])
            if "nodes" in graph:
                graph["nodes"] = normalise_str_list(graph["nodes"])
            normalised["module_dependency_graph"] = drop_extra_fields(
                graph,
                GraphDependencySpec,
            )

        if "repository_structure" in normalised:
            normalised["repository_structure"] = normalise_repository_structure(
                normalised["repository_structure"]
            )

        if "implementation_plan" in normalised:
            normalised["implementation_plan"] = normalise_items(
                normalised["implementation_plan"],
                ImplementationStep,
                fallback_field="action",
            )

        if "test_plan" in normalised:
            normalised["test_plan"] = normalise_test_plan(normalised["test_plan"])

        if "limitations" in normalised:
            normalised["limitations"] = normalise_str_list(normalised["limitations"])

        for field_name, model_cls in (
            ("generated_files", GeneratedFile),
            ("execution_results", ExecutionResult),
            ("tool_calls", ToolCallRecord),
        ):
            if field_name in normalised:
                normalised[field_name] = normalise_items(
                    normalised[field_name],
                    model_cls,
                    fallback_field="summary",
                )

        normalised = drop_extra_fields(normalised, cls)

        return normalised

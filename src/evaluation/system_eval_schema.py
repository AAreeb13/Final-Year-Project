from __future__ import annotations

from typing import Optional

from pydantic import Field, model_validator

from src.evaluation.sdlc_eval_schema import (
    ComponentSpec,
    GraphDependencySpec,
    HLArchitectureSpec,
    ImplementationStep,
    ModuleSpec,
    RepositoryStructure,
    RequirementsSpec,
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
            if isinstance(value, str):
                return [
                    item.strip()
                    for item in value.split(",")
                    if item.strip()
                ]
            return value

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
                normalised_signatures.append(normalised_signature)

            return normalised_signatures

        def normalise_module(module):
            if not isinstance(module, dict):
                return module

            normalised_module = dict(module)
            if "signatures" in normalised_module:
                normalised_module["signatures"] = normalise_signature_inputs(
                    normalised_module["signatures"]
                )
            return normalised_module

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
            normalised["requirements"] = requirements

        high_level_design = dict(normalised.get("high_level_design") or {})
        if "relationships" in normalised:
            high_level_design.setdefault(
                "relationships",
                normalised.pop("relationships") or [],
            )
        if high_level_design:
            normalised["high_level_design"] = high_level_design

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
                cleaned_components.append(component)
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

            cleaned_component = {
                key: value
                for key, value in component.items()
                if key in component_fields
            }
            if "technologies" in cleaned_component:
                cleaned_component["technologies"] = normalise_str_list(
                    cleaned_component["technologies"]
                )
            if cleaned_component:
                cleaned_components.append(cleaned_component)

        normalised["components"] = cleaned_components
        normalised["modules"] = modules

        return normalised

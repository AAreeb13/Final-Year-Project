from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


# ---------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------

class RequirementsSpec(StrictBaseModel):
    functional: list[str] = Field(default_factory=list)
    non_functional: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------
# High-level architecture
# ---------------------------------------------------------------------

class HLComponentSpec(StrictBaseModel):
    name: str = ""
    type: str = ""
    responsibilities: list[str] = Field(default_factory=list)


class RelationshipSpec(StrictBaseModel):
    source: str = ""
    target: str = ""
    relationship_type: str = ""


class HLArchitectureSpec(StrictBaseModel):
    style: Optional[str] = None
    components: list[HLComponentSpec] = Field(default_factory=list)
    relationships: list[RelationshipSpec] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------

class ComponentSpec(StrictBaseModel):
    name: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Modules and signatures
# ---------------------------------------------------------------------

SignatureType = Literal[
    "function",
    "class",
    "method",
    "schema",
    "interface",
    "api_endpoint",
]


class ParamSpec(StrictBaseModel):
    name: str = ""
    type: str = ""
    required: bool = True
    default: Optional[str] = None
    description: Optional[str] = None


class SignatureSpec(StrictBaseModel):
    type: SignatureType = "function"
    name: str = ""
    inputs: list[ParamSpec] = Field(default_factory=list)
    output: str = ""
    description: str = ""
    belongs_to: Optional[str] = None


class ModuleSpec(StrictBaseModel):
    name: str = ""
    component: str = ""
    type: Optional[str] = None
    responsibilities: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    signatures: list[SignatureSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Dependency graph
# ---------------------------------------------------------------------

class GraphDependencyEdgeSpec(StrictBaseModel):
    source: str = ""
    target: str = ""


class GraphDependencySpec(StrictBaseModel):
    nodes: list[str] = Field(default_factory=list)
    edges: list[GraphDependencyEdgeSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Repository structure
# ---------------------------------------------------------------------

class FileSpec(StrictBaseModel):
    name: str = ""
    modules: list[str] = Field(default_factory=list)


class DirectorySpec(StrictBaseModel):
    name: str = ""
    parent: Optional[str] = None
    children: list[str] = Field(default_factory=list)
    files: list[FileSpec] = Field(default_factory=list)


class RepositoryStructure(StrictBaseModel):
    directories: list[DirectorySpec] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Implementation and testing
# ---------------------------------------------------------------------

class ImplementationStep(StrictBaseModel):
    step_id: str = ""
    module_target: str = ""
    action: str = ""
    result: str = ""
    command: Optional[str] = None
    result_summary: Optional[str] = None


class TestPlan(StrictBaseModel):
    unit_tests: list[str] = Field(default_factory=list)
    integration_tests: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    testing_framework: str = ""

# ---------------------------------------------------------------------
# Project Information specification
# ---------------------------------------------------------------------

class ProjectInfoSpec(StrictBaseModel):
    project_id: str = ""
    project_prompt: str = ""
    project_type: str = ""
    difficulty: str = ""


# ---------------------------------------------------------------------
# Dataset project specification
# ---------------------------------------------------------------------

class ProjectSpec(StrictBaseModel):
    project_id: str
    project_prompt: str
    project_type: str = ""
    difficulty: str = ""

    requirements: RequirementsSpec = Field(default_factory=RequirementsSpec)
    high_level_design: HLArchitectureSpec = Field(default_factory=HLArchitectureSpec)

    components: list[ComponentSpec] = Field(default_factory=list)
    modules: list[ModuleSpec] = Field(default_factory=list)
    module_dependency_graph: GraphDependencySpec = Field(default_factory=GraphDependencySpec)

    repository_structure: RepositoryStructure = Field(default_factory=RepositoryStructure)
    implementation_plan: list[ImplementationStep] = Field(default_factory=list)
    test_plan: TestPlan = Field(default_factory=TestPlan)

    def __str__(self):
        # Create a human-readable string representation of the project specification
        return (f"Project ID: {self.project_id}\n"
                f"Prompt: {self.project_prompt}\n"
                f"Type: {self.project_type}\n"
                f"Difficulty: {self.difficulty}\n"
                f"Requirements: {len(self.requirements.functional) + len(self.requirements.non_functional)} total\n"
                f"Components: {len(self.components)}\n"
                f"Modules: {len(self.modules)}\n"
                f"Repository Directories: {len(self.repository_structure.directories)}\n"
                f"Implementation Steps: {len(self.implementation_plan)}\n"
                f"Test Plan: {len(self.test_plan.unit_tests) + len(self.test_plan.integration_tests)} tests\n"
        )
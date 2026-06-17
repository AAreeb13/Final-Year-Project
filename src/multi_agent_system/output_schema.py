from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class RequirementsSpec(BaseModel):
    functional_requirements: list[str] = Field(default_factory=list)
    non_functional_requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    def __str__(self):
        return (
            f"Functional Requirements:\n- " + "\n- ".join(self.functional_requirements) + "\n\n" +
            f"Non-Functional Requirements:\n- " + "\n- ".join(self.non_functional_requirements) + "\n\n" +
            f"Constraints:\n- " + "\n- ".join(self.constraints) + "\n\n" +
            f"Assumptions:\n- " + "\n- ".join(self.assumptions) + "\n\n" +
            f"Out of Scope:\n- " + "\n- ".join(self.out_of_scope)
        )

class QuestionAnswer(BaseModel):
    question: str
    answer: str
    def __str__(self):        return f"Q: {self.question}\nA: {self.answer}"


class RequirementsExtractorInput(BaseModel):
    project_prompt: str
    previous_requirements: RequirementsSpec | None = None
    question_answer_context: list[QuestionAnswer] = Field(default_factory=list)
    def __str__(self):        return (
            f"Project Prompt: {self.project_prompt}\n\n" +
            f"Previous Requirements: \n{self.previous_requirements}\n\n" +
            f"Question-Answer Context: {self.question_answer_context}"
        )


class RequirementsExtractorOutput(BaseModel):
    requirements: RequirementsSpec
    notes: list[str] = Field(default_factory=list)
    def __str__(self):        
        return (
            f"Extracted Requirements: \n{self.requirements}\n\n" +
            f"Extractor Notes: \n- " + "\n- ".join(self.notes)
        )


class RequirementsCriticInput(BaseModel):
    project_prompt: str
    requirements: RequirementsSpec
    question_answer_context: list[QuestionAnswer] = Field(default_factory=list)
    def __str__(self):        
        return (
            f"Project Prompt: {self.project_prompt}\n\n" +
            f"Requirements to Critique: \n{self.requirements}\n\n" +
            f"Question-Answer Context: {self.question_answer_context}"
    )


class RequirementsCriticOutput(BaseModel):
    approved: bool = False
    verdict: str = ""
    questions: list[str] = Field(default_factory=list)
    feedback: str = ""
    def __str__(self):
        return (
            f"Approved: {self.approved}\n\n" +
            f"Verdict: {self.verdict}\n\n" +
            f"Questions from Critic: \n- " + "\n- ".join(self.questions) + "\n\n" +
            f"Critic Feedback: {self.feedback}"
        )

class RequirementsStageOutput(BaseModel):
    requirements: RequirementsSpec
    question_answer_context: list[QuestionAnswer] = Field(default_factory=list)
    critic_verdict: str
    approved: bool
    def __str__(self):
        return (
            f"Requirements: \n{self.requirements}\n\n" +
            f"Question-Answer Context: \n{self.question_answer_context}\n\n" +
            f"Critic Verdict: {self.critic_verdict}\n\n" +
            f"Approved: {self.approved}"
        )


DesignTaskKind = Literal[
    "extract_components",
    "decompose_component",
    "design_modules",
]


class DesignTask(BaseModel):
    kind: DesignTaskKind
    target_component: str | None = None
    target_module: str | None = None

    @model_validator(mode="after")
    def validate_target_component(self):
        if self.kind in {"decompose_component", "design_modules"} and not self.target_component:
            raise ValueError(f"target_component is required for {self.kind}")
        return self


class HLComponentSpec(BaseModel):
    name: str = ""
    type: str = ""
    responsibilities: list[str] = Field(default_factory=list)


class RelationshipSpec(BaseModel):
    source: str = ""
    target: str = ""
    relationship_type: str = ""


class HLArchitectureSpec(BaseModel):
    style: str | None = None
    components: list[HLComponentSpec] = Field(default_factory=list)
    relationships: list[RelationshipSpec] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


class ComponentSpec(BaseModel):
    name: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


SignatureType = Literal[
    "function",
    "class",
    "method",
    "schema",
    "interface",
    "api_endpoint",
]


class ParamSpec(BaseModel):
    name: str = ""
    type: str = ""
    required: bool = True
    default: str | None = None
    description: str | None = None


class SignatureSpec(BaseModel):
    type: SignatureType = "function"
    name: str = ""
    inputs: list[ParamSpec] = Field(default_factory=list)
    output: str = ""
    description: str = ""
    belongs_to: str | None = None


class ModuleSpec(BaseModel):
    name: str = ""
    component: str = ""
    type: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    signatures: list[SignatureSpec] = Field(default_factory=list)


class GraphDependencyEdgeSpec(BaseModel):
    source: str = ""
    target: str = ""


class GraphDependencySpec(BaseModel):
    nodes: list[str] = Field(default_factory=list)
    edges: list[GraphDependencyEdgeSpec] = Field(default_factory=list)


class ComponentExtractionOutput(BaseModel):
    high_level_architecture: HLArchitectureSpec = Field(default_factory=HLArchitectureSpec)
    components: list[ComponentSpec] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ComponentDecompositionOutput(BaseModel):
    component: ComponentSpec = Field(default_factory=ComponentSpec)
    modules: list[ModuleSpec] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ModuleDesignOutput(BaseModel):
    component_name: str = ""
    modules: list[ModuleSpec] = Field(default_factory=list)
    dependency_graph: GraphDependencySpec = Field(default_factory=GraphDependencySpec)
    notes: list[str] = Field(default_factory=list)


class FileSpec(BaseModel):
    name: str = ""
    modules: list[str] = Field(default_factory=list)


class DirectorySpec(BaseModel):
    name: str = ""
    parent: str | None = None
    children: list[str] = Field(default_factory=list)
    files: list[FileSpec] = Field(default_factory=list)


class RepositoryStructure(BaseModel):
    directories: list[DirectorySpec] = Field(default_factory=list)


class SetupFileSpec(BaseModel):
    path: str = ""
    content: str = ""
    description: str = ""


class EnvironmentSetupPlan(BaseModel):
    language: str = ""
    framework: str = ""
    package_manager: str = ""
    dependency_files: list[str] = Field(default_factory=list)
    setup_files: list[SetupFileSpec] = Field(default_factory=list)
    setup_commands: list[list[str]] = Field(default_factory=list)
    test_commands: list[list[str]] = Field(default_factory=list)
    build_commands: list[list[str]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("setup_commands", "test_commands", "build_commands", mode="before")
    @classmethod
    def normalize_commands(cls, value: Any) -> Any:
        return _normalize_command_list(value)


class EnvironmentSetupStageOutput(BaseModel):
    repository_structure: RepositoryStructure = Field(default_factory=RepositoryStructure)
    environment_setup: EnvironmentSetupPlan = Field(default_factory=EnvironmentSetupPlan)
    approved: bool = False
    approval_feedback: str = ""
    notes: list[str] = Field(default_factory=list)


class SetupExecutionResult(BaseModel):
    step: str = ""
    tool_name: str = ""
    command: list[str] = Field(default_factory=list)
    path: str | None = None
    status: str = "unknown"
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    message: str = ""
    raw_output: str = ""


class EnvironmentSetupExecution(BaseModel):
    results: list[SetupExecutionResult] = Field(default_factory=list)
    status: str = "not_started"
    summary: str = ""


class PlanningGraphInput(BaseModel):
    project_prompt: str
    requirements: RequirementsSpec
    component_extraction: ComponentExtractionOutput
    component_decompositions: dict[str, ComponentDecompositionOutput] = Field(default_factory=dict)
    module_designs: dict[str, ModuleDesignOutput] = Field(default_factory=dict)
    repository_structure: RepositoryStructure | None = None
    environment_setup: EnvironmentSetupPlan | None = None


class ImplementationFilePlan(BaseModel):
    file_id: str
    relative_path: str
    component_name: str
    modules: list[str] = Field(default_factory=list)
    purpose: str
    unit_tests: list[str] = Field(default_factory=list)


class ImplementationPlanStep(BaseModel):
    step_id: str
    action: str
    target_file_id: str | None = None
    target_modules: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    description: str = ""


class ImplementationPlan(BaseModel):
    files: list[ImplementationFilePlan] = Field(default_factory=list)
    steps: list[ImplementationPlanStep] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_plan_references(self):
        file_ids = [file.file_id for file in self.files]
        file_id_set = set(file_ids)
        if len(file_ids) != len(file_id_set):
            raise ValueError("Implementation file ids must be unique")

        step_ids = [step.step_id for step in self.steps]
        step_id_set = set(step_ids)
        if len(step_ids) != len(step_id_set):
            raise ValueError("Implementation step ids must be unique")

        for step in self.steps:
            if step.target_file_id and step.target_file_id not in file_id_set:
                raise ValueError(f"Implementation step {step.step_id} targets unknown file_id")
            missing_dependencies = [
                dependency_id
                for dependency_id in step.depends_on
                if dependency_id not in step_id_set
            ]
            if missing_dependencies:
                raise ValueError(
                    f"Implementation step {step.step_id} depends on unknown steps: {missing_dependencies}"
                )
        return self


class TestCaseSpec(BaseModel):
    name: str = ""
    target_signature: str | None = None
    input_summary: str = ""
    expected_output_summary: str = ""
    assertion_summary: str = ""


TestPlanNodeKind = Literal[
    "unit",
    "module_integration",
    "component_integration",
    "system_integration",
]


class TestPlanNode(BaseModel):
    node_id: str
    kind: TestPlanNodeKind
    title: str
    target_modules: list[str] = Field(default_factory=list)
    target_signatures: list[str] = Field(default_factory=list)
    test_cases: list[TestCaseSpec] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("test_cases", mode="before")
    @classmethod
    def normalize_test_cases(cls, value: Any) -> Any:
        if value in (None, ""):
            return []
        if not isinstance(value, list):
            return value
        normalized = []
        for item in value:
            if isinstance(item, str):
                normalized.append(
                    {
                        "name": item,
                        "input_summary": "Use representative valid and invalid inputs for this case.",
                        "expected_output_summary": item,
                        "assertion_summary": item,
                    }
                )
                continue
            normalized.append(item)
        return normalized


class TestPlanEdge(BaseModel):
    source: str
    target: str
    relationship_type: str = "depends_on"


class TestPlanGraph(BaseModel):
    testing_framework: str = ""
    nodes: list[TestPlanNode] = Field(default_factory=list)
    edges: list[TestPlanEdge] = Field(default_factory=list)
    root_node_id: str = ""
    commands: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph_references(self):
        node_ids = [node.node_id for node in self.nodes]
        node_id_set = set(node_ids)
        if len(node_ids) != len(node_id_set):
            raise ValueError("Test plan node ids must be unique")
        if self.root_node_id and self.root_node_id not in node_id_set:
            raise ValueError("root_node_id must reference an existing test plan node")
        for node in self.nodes:
            missing_dependencies = [
                dependency_id
                for dependency_id in node.depends_on
                if dependency_id not in node_id_set
            ]
            if missing_dependencies:
                raise ValueError(
                    f"Test plan node {node.node_id} depends on unknown nodes: {missing_dependencies}"
                )
        for edge in self.edges:
            if edge.source not in node_id_set or edge.target not in node_id_set:
                raise ValueError("Test plan edges must reference existing node ids")
        return self


class PlanningStageOutput(BaseModel):
    implementation_plan: ImplementationPlan = Field(default_factory=ImplementationPlan)
    test_plan: TestPlanGraph = Field(default_factory=TestPlanGraph)
    critic_verdict: str = ""
    validation_errors: list[str] = Field(default_factory=list)
    approved: bool = False
    notes: list[str] = Field(default_factory=list)


class FileWriteResult(BaseModel):
    relative_path: str = ""
    status: str = "unknown"
    message: str = ""
    summary: str = ""


class TestExecutionResult(BaseModel):
    command: list[str] = Field(default_factory=list)
    status: str = "unknown"
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    message: str = ""
    summary: str = ""


class ImplementationExecutionResult(BaseModel):
    file_writes: list[FileWriteResult] = Field(default_factory=list)
    test_executions: list[TestExecutionResult] = Field(default_factory=list)
    status: str = "not_started"
    summary: str = ""


class ImplementationStageOutput(BaseModel):
    execution: ImplementationExecutionResult = Field(default_factory=ImplementationExecutionResult)
    approved: bool = False
    failure_reason: str = ""
    notes: list[str] = Field(default_factory=list)


TestCaseSpec.__test__ = False
TestPlanNode.__test__ = False
TestPlanEdge.__test__ = False
TestPlanGraph.__test__ = False


PlanningTaskKind = Literal[
    "create_file",
    "create_test_file",
    "execute_test",
    "create_integration_test",
    "execute_integration_test",
    "review",
]


class PlanningTask(BaseModel):
    task_id: str
    kind: PlanningTaskKind
    relative_path: str | None = None
    target_component: str
    target_module: str | None = None
    description: str
    depends_on: list[str] = Field(default_factory=list)
    test_target_task_id: str | None = None


class ModulePlanNode(BaseModel):
    module_name: str
    implementation_task_id: str
    test_task_id: str
    execute_test_task_id: str
    notes: list[str] = Field(default_factory=list)


class ComponentImplementationPlan(BaseModel):
    component_name: str
    summary: str
    module_nodes: list[ModulePlanNode] = Field(default_factory=list)
    integration_test_task_id: str | None = None
    execute_integration_test_task_id: str | None = None
    tasks: list[PlanningTask] = Field(default_factory=list)
    testing_strategy: str = ""
    approved: bool = False

    @model_validator(mode="after")
    def validate_task_tree(self):
        task_ids = [task.task_id for task in self.tasks]
        task_id_set = set(task_ids)
        if len(task_ids) != len(task_id_set):
            raise ValueError("Planning task ids must be unique")

        for task in self.tasks:
            missing_dependencies = [task_id for task_id in task.depends_on if task_id not in task_id_set]
            if missing_dependencies:
                raise ValueError(
                    f"Planning task {task.task_id} depends on unknown tasks: {missing_dependencies}"
                )
            if task.test_target_task_id is not None and task.test_target_task_id not in task_id_set:
                raise ValueError(
                    f"Planning task {task.task_id} targets unknown test task: {task.test_target_task_id}"
                )

        for node in self.module_nodes:
            for field_name in ("implementation_task_id", "test_task_id", "execute_test_task_id"):
                task_id = getattr(node, field_name)
                if task_id not in task_id_set:
                    raise ValueError(f"Module node {node.module_name} references unknown task: {task_id}")

        if self.integration_test_task_id and self.integration_test_task_id not in task_id_set:
            raise ValueError("integration_test_task_id references an unknown task")
        if self.execute_integration_test_task_id and self.execute_integration_test_task_id not in task_id_set:
            raise ValueError("execute_integration_test_task_id references an unknown task")
        return self


def _normalize_command_list(value: Any) -> Any:
    if value in (None, ""):
        return []
    if isinstance(value, tuple):
        return [list(value)]
    if isinstance(value, str):
        return [value.split()]
    if not isinstance(value, list):
        return value

    normalized = []
    for item in value:
        if item in (None, ""):
            continue
        if isinstance(item, str):
            normalized.append(item.split())
            continue
        if isinstance(item, tuple):
            normalized.append([str(part) for part in item])
            continue
        if isinstance(item, list):
            normalized.append([str(part) for part in item])
            continue
        normalized.append(item)
    return normalized


class ArchitectInput(BaseModel):
    project_prompt: str
    requirements: RequirementsStageOutput
    task: DesignTask
    previous_design_output: "DesignStageOutput | None" = None
    question_answer_context: list[QuestionAnswer] = Field(default_factory=list)
    critic_feedback: str = ""


class ArchitectOutput(BaseModel):
    task: DesignTask | None = None
    component_extraction: ComponentExtractionOutput | None = None
    component_decomposition: ComponentDecompositionOutput | None = None
    module_design: ModuleDesignOutput | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_matching_artifact(self):
        artifact_fields = {
            "extract_components": self.component_extraction,
            "decompose_component": self.component_decomposition,
            "design_modules": self.module_design,
        }
        populated_count = sum(artifact is not None for artifact in artifact_fields.values())
        if populated_count != 1:
            raise ValueError("ArchitectOutput must populate exactly one design artifact field")
        if self.task is None:
            return self
        expected_artifact = artifact_fields[self.task.kind]
        if expected_artifact is None:
            raise ValueError(f"{self.task.kind} output must populate its matching artifact field")
        return self


class DesignCriticInput(BaseModel):
    project_prompt: str
    requirements: RequirementsStageOutput
    task: DesignTask
    architect_output: ArchitectOutput
    question_answer_context: list[QuestionAnswer] = Field(default_factory=list)


class DesignCriticOutput(BaseModel):
    approved: bool = False
    verdict: str = ""
    questions: list[str] = Field(default_factory=list)
    feedback: str = ""
    required_changes: list[str] = Field(default_factory=list)


class DesignStageOutput(BaseModel):
    task: DesignTask
    architect_output: ArchitectOutput
    question_answer_context: list[QuestionAnswer] = Field(default_factory=list)
    critic_verdict: str
    approved: bool

    @property
    def component_extraction(self) -> ComponentExtractionOutput | None:
        return self.architect_output.component_extraction

    @property
    def component_decomposition(self) -> ComponentDecompositionOutput | None:
        return self.architect_output.component_decomposition

    @property
    def module_design(self) -> ModuleDesignOutput | None:
        return self.architect_output.module_design


class ProjectStore(BaseModel):
    project_id: str
    project_prompt: str
    requirements: RequirementsStageOutput | None = None
    component_extraction: ComponentExtractionOutput | None = None
    component_decompositions: dict[str, ComponentDecompositionOutput] = Field(default_factory=dict)
    module_designs: dict[str, ModuleDesignOutput] = Field(default_factory=dict)
    repository_structure: RepositoryStructure | None = None
    environment_setup: EnvironmentSetupPlan | None = None
    environment_setup_execution: EnvironmentSetupExecution | None = None
    planning: PlanningStageOutput | None = None
    implementation: ImplementationStageOutput | None = None
    stage_statuses: dict[str, str] = Field(default_factory=dict)

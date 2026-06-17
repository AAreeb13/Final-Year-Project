from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from src.agents.AgentFactory import AgentFactory
from src.multi_agent_system.output_schema import (
    ComponentDecompositionOutput,
    ComponentExtractionOutput,
    EnvironmentSetupPlan,
    GraphDependencyEdgeSpec,
    ImplementationPlan,
    ModuleDesignOutput,
    ModuleSpec,
    PlanningGraphInput,
    PlanningStageOutput,
    RepositoryStructure,
    RequirementsSpec,
    TestPlanGraph,
)
from src.multi_agent_system.store import ProjectStoreRepository
from src.settings import configure_langsmith_environment, settings


SCHEMA_FAILURE_LOG_PATH = Path(__file__).with_name("planning_schema_failures.txt")

PlanningStatus = Literal["complete", "needs_revision", "failed"]
AgentRunner = Callable[[Any, BaseModel], Any]
ModelT = TypeVar("ModelT", bound=BaseModel)


class FormatErrorDecision(BaseModel):
    retry: bool = True
    feedback: str = ""


FormatErrorCallback = Callable[[str, BaseModel, Exception, int, int], FormatErrorDecision]


IMPLEMENTATION_PLANNER_PROMPT = """
You are the Implementation Planner in an artifact-only SDLC planning graph.

Create an ImplementationPlan that maps every provided module to one or more files.
For each file:
- include the component name
- list the modules written in that file
- describe the file purpose
- include simple unit test strings for the modules in that file

Rules:
- Return planning artifacts only. Do not write code and do not include shell execution tasks.
- Cover every module in planning_context.module_names.
- Keep unit_tests as plain strings.
- Use previous_planning_feedback as mandatory revision guidance when present.
""".strip()


TEST_TREE_PLANNER_PROMPT = """
You are the Test Tree Planner in an artifact-only SDLC planning graph.

Create a TestPlanGraph that grows from unit tests to integration tests:
- unit nodes test module signatures
- module_integration nodes test directly related modules together
- component_integration nodes test modules inside each component
- one system_integration root node tests the whole system

Rules:
- Cover every signature in planning_context.signature_ids with unit test nodes.
- Add integration coverage for planning_context.dependency_edges.
- root_node_id must point to the single system_integration node.
- Use depends_on and edges to make the tree unambiguous.
- Use previous_planning_feedback as mandatory revision guidance when present.
""".strip()


PLANNING_CRITIC_PROMPT = """
You are the Planning Critic for a multi-agent SDLC planning phase.

Review the ImplementationPlan and TestPlanGraph for:
- complete module coverage
- useful file/module mapping
- signature-level unit coverage
- integration coverage across module relationships
- agreement between implementation files and test graph targets
- no Q&A context leakage

If deterministic validation errors are present, set approved=false and include them in required_changes.
Approve only when both artifacts are specific enough for a later implementation graph.
""".strip()


class PlanningModuleContext(BaseModel):
    component_name: str = ""
    module_name: str = ""
    dependencies: list[str] = Field(default_factory=list)
    signature_ids: list[str] = Field(default_factory=list)


class PlanningDependencyContext(BaseModel):
    source: str = ""
    target: str = ""
    component_name: str = ""


class PlanningContext(BaseModel):
    module_names: list[str] = Field(default_factory=list)
    modules: list[PlanningModuleContext] = Field(default_factory=list)
    signature_ids: list[str] = Field(default_factory=list)
    dependency_edges: list[PlanningDependencyContext] = Field(default_factory=list)


class ImplementationPlannerInput(BaseModel):
    project_prompt: str
    requirements: RequirementsSpec
    component_extraction: ComponentExtractionOutput
    component_decompositions: dict[str, ComponentDecompositionOutput] = Field(default_factory=dict)
    module_designs: dict[str, ModuleDesignOutput] = Field(default_factory=dict)
    repository_structure: RepositoryStructure | None = None
    environment_setup: EnvironmentSetupPlan | None = None
    planning_context: PlanningContext
    previous_planning_feedback: str = ""


class TestTreePlannerInput(BaseModel):
    project_prompt: str
    requirements: RequirementsSpec
    component_extraction: ComponentExtractionOutput
    component_decompositions: dict[str, ComponentDecompositionOutput] = Field(default_factory=dict)
    module_designs: dict[str, ModuleDesignOutput] = Field(default_factory=dict)
    implementation_plan: ImplementationPlan
    planning_context: PlanningContext
    previous_planning_feedback: str = ""


class PlanningCriticInput(BaseModel):
    project_prompt: str
    requirements: RequirementsSpec
    implementation_plan: ImplementationPlan
    test_plan: TestPlanGraph
    planning_context: PlanningContext
    validation_errors: list[str] = Field(default_factory=list)
    format_feedback: str = ""


class PlanningCriticOutput(BaseModel):
    approved: bool = False
    verdict: str = ""
    feedback: str = ""
    required_changes: list[str] = Field(default_factory=list)


class PlanningState(TypedDict, total=False):
    graph_input: PlanningGraphInput
    planning_context: PlanningContext
    previous_planning_feedback: str
    implementation_plan: ImplementationPlan
    test_plan: TestPlanGraph
    validation_errors: list[str]
    critic_output: PlanningCriticOutput
    stage_output: PlanningStageOutput
    revision_count: int
    max_iterations: int
    status: PlanningStatus
    failure_reason: str


class PlanningRunResult(BaseModel):
    status: PlanningStatus
    stage_output: PlanningStageOutput | None = None
    implementation_plan: ImplementationPlan | None = None
    test_plan: TestPlanGraph | None = None
    critic_output: PlanningCriticOutput | None = None
    validation_errors: list[str] = Field(default_factory=list)
    revision_count: int = 0
    failure_reason: str = ""


class PlanningGraph:
    """Artifact-only planning graph for implementation and test-plan outputs."""

    def __init__(
        self,
        implementation_planner_agent: Any | None = None,
        test_tree_planner_agent: Any | None = None,
        critic_agent: Any | None = None,
        implementation_planner_runner: AgentRunner | None = None,
        test_tree_planner_runner: AgentRunner | None = None,
        critic_runner: AgentRunner | None = None,
        format_error_callback: FormatErrorCallback | None = None,
        max_format_retries: int = 2,
        verbose: bool = True,
        model_name: str = "gpt-5.4-mini",
        temperature: float = 0.2,
    ) -> None:
        self.implementation_planner_agent = implementation_planner_agent
        self.test_tree_planner_agent = test_tree_planner_agent
        self.critic_agent = critic_agent
        self.implementation_planner_runner = implementation_planner_runner or _run_structured_agent
        self.test_tree_planner_runner = test_tree_planner_runner or _run_structured_agent
        self.critic_runner = critic_runner or _run_structured_agent
        self.format_error_callback = format_error_callback
        self.max_format_retries = max(0, max_format_retries)
        self.verbose = verbose
        self.model_name = model_name
        self.temperature = temperature
        self.graph = self._build_graph()

    def run(
        self,
        project_prompt: str,
        requirements: RequirementsSpec,
        component_extraction: ComponentExtractionOutput,
        component_decompositions: dict[str, ComponentDecompositionOutput],
        module_designs: dict[str, ModuleDesignOutput],
        repository_structure: RepositoryStructure | None = None,
        environment_setup: EnvironmentSetupPlan | None = None,
        previous_planning_feedback: str = "",
        max_iterations: int = 3,
    ) -> PlanningRunResult:
        configure_langsmith_environment(settings)
        graph_input = PlanningGraphInput(
            project_prompt=project_prompt,
            requirements=requirements,
            component_extraction=component_extraction,
            component_decompositions=component_decompositions,
            module_designs=module_designs,
            repository_structure=repository_structure,
            environment_setup=environment_setup,
        )
        initial_state: PlanningState = {
            "graph_input": graph_input,
            "previous_planning_feedback": previous_planning_feedback,
            "revision_count": 0,
            "max_iterations": max_iterations,
        }
        try:
            final_state = self.graph.invoke(initial_state)
            return self._to_result(final_state)
        except Exception as error:
            return PlanningRunResult(
                status="failed",
                failure_reason=_format_exception(error),
            )

    def _build_graph(self):
        graph_builder = StateGraph(PlanningState)
        graph_builder.add_node("prepare_context", self.prepare_context)
        graph_builder.add_node("implementation_planner", self.implementation_planner)
        graph_builder.add_node("test_tree_planner", self.test_tree_planner)
        graph_builder.add_node("deterministic_validator", self.deterministic_validator)
        graph_builder.add_node("critic_review", self.critic_review)
        graph_builder.add_node("complete", self.complete)
        graph_builder.add_node("failed", self.failed)

        graph_builder.add_edge(START, "prepare_context")
        graph_builder.add_edge("prepare_context", "implementation_planner")
        graph_builder.add_edge("implementation_planner", "test_tree_planner")
        graph_builder.add_edge("test_tree_planner", "deterministic_validator")
        graph_builder.add_edge("deterministic_validator", "critic_review")
        graph_builder.add_conditional_edges(
            "critic_review",
            self.route_after_critic,
            {
                "complete": "complete",
                "implementation_planner": "implementation_planner",
                "failed": "failed",
            },
        )
        graph_builder.add_edge("complete", END)
        graph_builder.add_edge("failed", END)
        return graph_builder.compile()

    def prepare_context(self, state: PlanningState) -> PlanningState:
        self._announce_node("prepare_context")
        return {
            **state,
            "planning_context": _build_planning_context(state["graph_input"].module_designs),
        }

    def implementation_planner(self, state: PlanningState) -> PlanningState:
        self._announce_node("implementation_planner")
        graph_input = state["graph_input"]
        planner_input = ImplementationPlannerInput(
            project_prompt=graph_input.project_prompt,
            requirements=graph_input.requirements,
            component_extraction=graph_input.component_extraction,
            component_decompositions=graph_input.component_decompositions,
            module_designs=graph_input.module_designs,
            repository_structure=graph_input.repository_structure,
            environment_setup=graph_input.environment_setup,
            planning_context=state["planning_context"],
            previous_planning_feedback=state.get("previous_planning_feedback", ""),
        )
        return {
            **state,
            "implementation_plan": self._run_agent_with_format_retries(
                node_name="implementation_planner",
                agent=self._get_implementation_planner_agent(),
                agent_input=planner_input,
                runner=self.implementation_planner_runner,
                model_cls=ImplementationPlan,
            ),
        }

    def test_tree_planner(self, state: PlanningState) -> PlanningState:
        self._announce_node("test_tree_planner")
        graph_input = state["graph_input"]
        planner_input = TestTreePlannerInput(
            project_prompt=graph_input.project_prompt,
            requirements=graph_input.requirements,
            component_extraction=graph_input.component_extraction,
            component_decompositions=graph_input.component_decompositions,
            module_designs=graph_input.module_designs,
            implementation_plan=state["implementation_plan"],
            planning_context=state["planning_context"],
            previous_planning_feedback=state.get("previous_planning_feedback", ""),
        )
        return {
            **state,
            "test_plan": self._run_agent_with_format_retries(
                node_name="test_tree_planner",
                agent=self._get_test_tree_planner_agent(),
                agent_input=planner_input,
                runner=self.test_tree_planner_runner,
                model_cls=TestPlanGraph,
            ),
        }

    def deterministic_validator(self, state: PlanningState) -> PlanningState:
        self._announce_node("deterministic_validator")
        validation_errors = _validate_planning_artifacts(
            state["implementation_plan"],
            state["test_plan"],
            state["planning_context"],
        )
        return {
            **state,
            "validation_errors": validation_errors,
        }

    def critic_review(self, state: PlanningState) -> PlanningState:
        self._announce_node("critic_review")
        graph_input = state["graph_input"]
        critic_input = PlanningCriticInput(
            project_prompt=graph_input.project_prompt,
            requirements=graph_input.requirements,
            implementation_plan=state["implementation_plan"],
            test_plan=state["test_plan"],
            planning_context=state["planning_context"],
            validation_errors=state.get("validation_errors", []),
        )
        critic_output = self._run_agent_with_format_retries(
            node_name="critic_review",
            agent=self._get_critic_agent(),
            agent_input=critic_input,
            runner=self.critic_runner,
            model_cls=PlanningCriticOutput,
        )

        if state.get("validation_errors") or not critic_output.approved:
            revision_count = state.get("revision_count", 0) + 1
            next_feedback = _append_feedback(
                state.get("previous_planning_feedback", ""),
                _format_revision_feedback(state.get("validation_errors", []), critic_output),
            )
            next_state: PlanningState = {
                **state,
                "critic_output": critic_output,
                "revision_count": revision_count,
                "previous_planning_feedback": next_feedback,
                "status": "needs_revision",
            }
            if revision_count >= state.get("max_iterations", 3):
                return {
                    **next_state,
                    "status": "failed",
                    "failure_reason": (
                        "Planning artifacts did not pass review before the iteration limit "
                        f"({revision_count}/{state.get('max_iterations', 3)})."
                    ),
                }
            return next_state

        return {
            **state,
            "critic_output": critic_output,
            "status": "complete",
        }

    @staticmethod
    def route_after_critic(state: PlanningState) -> str:
        if state.get("status") == "complete":
            print("Planning graph route: critic_review -> complete")
            return "complete"
        if state.get("status") == "failed":
            print("Planning graph route: critic_review -> failed")
            return "failed"
        print("Planning graph route: critic_review -> implementation_planner")
        return "implementation_planner"

    @staticmethod
    def complete(state: PlanningState) -> PlanningState:
        print("Planning graph node: complete")
        critic_output = state["critic_output"]
        stage_output = PlanningStageOutput(
            implementation_plan=state["implementation_plan"],
            test_plan=state["test_plan"],
            critic_verdict=critic_output.verdict,
            validation_errors=state.get("validation_errors", []),
            approved=True,
        )
        return {
            **state,
            "stage_output": stage_output,
            "status": "complete",
        }

    @staticmethod
    def failed(state: PlanningState) -> PlanningState:
        print("Planning graph node: failed")
        critic_output = state.get("critic_output")
        stage_output = None
        if "implementation_plan" in state and "test_plan" in state:
            stage_output = PlanningStageOutput(
                implementation_plan=state["implementation_plan"],
                test_plan=state["test_plan"],
                critic_verdict=critic_output.verdict if critic_output else "",
                validation_errors=state.get("validation_errors", []),
                approved=False,
            )
        return {
            **state,
            "stage_output": stage_output,
            "status": "failed",
            "failure_reason": state.get("failure_reason", "Planning graph failed."),
        }

    def _get_implementation_planner_agent(self) -> Any:
        if self.implementation_planner_agent is None:
            self.implementation_planner_agent = AgentFactory.build_agent(
                prompt=IMPLEMENTATION_PLANNER_PROMPT,
                tools=[],
                temperature=self.temperature,
                model_name=self.model_name,
                response_format=ImplementationPlan,
            )
        return self.implementation_planner_agent

    def _get_test_tree_planner_agent(self) -> Any:
        if self.test_tree_planner_agent is None:
            self.test_tree_planner_agent = AgentFactory.build_agent(
                prompt=TEST_TREE_PLANNER_PROMPT,
                tools=[],
                temperature=self.temperature,
                model_name=self.model_name,
                response_format=TestPlanGraph,
            )
        return self.test_tree_planner_agent

    def _get_critic_agent(self) -> Any:
        if self.critic_agent is None:
            self.critic_agent = AgentFactory.build_agent(
                prompt=PLANNING_CRITIC_PROMPT,
                tools=[],
                temperature=self.temperature,
                model_name=self.model_name,
                response_format=PlanningCriticOutput,
            )
        return self.critic_agent

    def _announce_node(self, node_name: str) -> None:
        if self.verbose:
            print(f"Planning graph node: {node_name}")

    def _run_agent_with_format_retries(
        self,
        node_name: str,
        agent: Any,
        agent_input: ModelT,
        runner: AgentRunner,
        model_cls: type[ModelT],
    ) -> ModelT:
        current_input = agent_input
        attempt = 0
        while True:
            try:
                raw_output = runner(agent, current_input)
                return _coerce_model(raw_output, model_cls)
            except Exception as error:
                if not _is_format_error(error):
                    raise
                attempt += 1
                if attempt > self.max_format_retries:
                    raise
                decision = self._handle_format_error(
                    node_name=node_name,
                    agent_input=current_input,
                    error=error,
                    attempt=attempt,
                )
                if not decision.retry:
                    raise
                current_input = _add_format_feedback(current_input, decision.feedback)

    def _handle_format_error(
        self,
        node_name: str,
        agent_input: ModelT,
        error: Exception,
        attempt: int,
    ) -> FormatErrorDecision:
        max_attempts = self.max_format_retries
        if self.format_error_callback is not None:
            return self.format_error_callback(
                node_name,
                agent_input,
                error,
                attempt,
                max_attempts,
            )

        return FormatErrorDecision(
            retry=False,
            feedback=(
                "Return a valid structured output for this node. "
                f"Error: {_format_exception(error)}"
            ),
        )

    @staticmethod
    def _to_result(state: PlanningState) -> PlanningRunResult:
        return PlanningRunResult(
            status=state.get("status", "failed"),
            stage_output=state.get("stage_output"),
            implementation_plan=state.get("implementation_plan"),
            test_plan=state.get("test_plan"),
            critic_output=state.get("critic_output"),
            validation_errors=state.get("validation_errors", []),
            revision_count=state.get("revision_count", 0),
            failure_reason=state.get("failure_reason", ""),
        )


def build_planning_graph(**kwargs: Any) -> PlanningGraph:
    return PlanningGraph(**kwargs)


def _load_planning_inputs_from_store(
    project_id: str,
    store_repository: ProjectStoreRepository | None = None,
) -> dict[str, Any]:
    repository = store_repository or ProjectStoreRepository()
    store = repository.load_project(project_id)

    if store.requirements is None:
        raise RuntimeError(f"Project {project_id} does not have approved requirements in the store.")
    if store.component_extraction is None:
        raise RuntimeError(f"Project {project_id} does not have component extraction in the store.")
    if not store.component_decompositions:
        raise RuntimeError(f"Project {project_id} does not have component decompositions in the store.")
    if not store.module_designs:
        raise RuntimeError(f"Project {project_id} does not have module designs in the store.")

    return {
        "project_id": store.project_id,
        "project_prompt": store.project_prompt,
        "requirements": store.requirements.requirements,
        "component_extraction": store.component_extraction,
        "component_decompositions": store.component_decompositions,
        "module_designs": store.module_designs,
        "repository_structure": store.repository_structure,
        "environment_setup": store.environment_setup,
    }


def _print_loaded_planning_inputs(inputs: dict[str, Any]) -> None:
    requirements = inputs["requirements"]
    component_extraction = inputs["component_extraction"]
    component_decompositions = inputs["component_decompositions"]
    module_designs = inputs["module_designs"]
    module_count = sum(len(module_design.modules) for module_design in module_designs.values())
    signature_count = sum(
        len(module.signatures)
        for module_design in module_designs.values()
        for module in module_design.modules
    )

    print("\nLoaded planning inputs")
    print(f"Workspace folder: {settings.WORKPLACE_FOLDER}")
    print(f"Project id: {inputs['project_id']}")
    print(f"Project prompt: {inputs['project_prompt']}")
    print(f"Functional requirements: {len(requirements.functional_requirements)}")
    print(f"Non-functional requirements: {len(requirements.non_functional_requirements)}")
    print(f"Components extracted: {len(component_extraction.components)}")
    print(f"Component decompositions: {len(component_decompositions)}")
    print(f"Module designs: {len(module_designs)}")
    print(f"Modules planned from design: {module_count}")
    print(f"Signatures planned from design: {signature_count}")
    print(f"Repository structure loaded: {inputs.get('repository_structure') is not None}")
    print(f"Environment setup loaded: {inputs.get('environment_setup') is not None}")


def _print_planning_result(result: PlanningRunResult) -> None:
    print("\nPlanning graph result")
    print(f"Status: {result.status}")
    print(f"Revision count: {result.revision_count}")
    if result.failure_reason:
        print(f"Failure reason: {result.failure_reason}")
    if result.validation_errors:
        print("Validation errors:")
        for error in result.validation_errors:
            print(f"- {error}")
    if result.critic_output is not None:
        print(f"Critic approved: {result.critic_output.approved}")
        if result.critic_output.verdict:
            print(f"Critic verdict: {result.critic_output.verdict}")
        if result.critic_output.required_changes:
            print("Required changes:")
            for change in result.critic_output.required_changes:
                print(f"- {change}")
    if result.stage_output is None:
        return

    implementation_plan = result.stage_output.implementation_plan
    test_plan = result.stage_output.test_plan
    print(f"Approved stage output: {result.stage_output.approved}")
    print(f"Implementation files: {len(implementation_plan.files)}")
    print(f"Implementation steps: {len(implementation_plan.steps)}")
    print(f"Test graph nodes: {len(test_plan.nodes)}")
    print(f"Test graph edges: {len(test_plan.edges)}")
    print(f"Test graph root: {test_plan.root_node_id or 'None'}")


def _interactive_format_error_decision(
    node_name: str,
    agent_input: BaseModel,
    error: Exception,
    attempt: int,
    max_attempts: int,
) -> FormatErrorDecision:
    _record_schema_failure(
        node_name=node_name,
        agent_input=agent_input,
        error=error,
        attempt=attempt,
        max_attempts=max_attempts,
    )
    print("\nStructured output error")
    print(f"Node: {node_name}")
    print(f"Attempt: {attempt}/{max_attempts}")
    print("Error:")
    print(_format_exception(error))
    invalid_output = _extract_invalid_output(error)
    if invalid_output is not None:
        print("\nInvalid output:")
        print(json.dumps(invalid_output, indent=2) if isinstance(invalid_output, dict) else invalid_output)
    print("\nInput fields sent to this node:")
    print(agent_input.model_dump_json(indent=2))

    choice = input("\nRetry this node with extra advice? [Y/n]: ").strip().lower()
    if choice in {"n", "no"}:
        return FormatErrorDecision(retry=False)

    feedback = input(
        "Advice for the agent to produce valid output "
        "(for example: make depends_on only reference existing node_id values): "
    ).strip()
    if not feedback:
        feedback = (
            "Regenerate a valid structured response. Make every ID reference point to an "
            "object that exists in the same output, and follow the schema exactly."
        )
    return FormatErrorDecision(retry=True, feedback=feedback)


def _record_schema_failure(
    node_name: str,
    agent_input: BaseModel,
    error: Exception,
    attempt: int,
    max_attempts: int,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    invalid_output = _extract_invalid_output(error)
    invalid_output_text = (
        json.dumps(invalid_output, indent=2)
        if isinstance(invalid_output, dict)
        else str(invalid_output)
        if invalid_output is not None
        else "None exposed by structured output error."
    )
    entry = (
        "\n"
        + "=" * 88
        + "\n"
        f"timestamp_utc: {timestamp}\n"
        f"node: {node_name}\n"
        f"attempt: {attempt}/{max_attempts}\n"
        "error:\n"
        f"{_format_exception(error)}\n\n"
        "input_fields:\n"
        f"{agent_input.model_dump_json(indent=2)}\n\n"
        "invalid_output:\n"
        f"{invalid_output_text}\n"
    )
    SCHEMA_FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCHEMA_FAILURE_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(entry)


def main() -> None:
    project_id = input("Project id to run planning graph [todo_cli]: ").strip() or "todo_cli"
    store_repository = ProjectStoreRepository()
    inputs = _load_planning_inputs_from_store(project_id, store_repository)
    _print_loaded_planning_inputs(inputs)

    choice = input("\nRun planning graph with these inputs? [y/N]: ").strip().lower()
    if choice not in {"y", "yes"}:
        return

    graph_inputs = {key: value for key, value in inputs.items() if key != "project_id"}
    result = build_planning_graph(
        format_error_callback=_interactive_format_error_decision,
        max_format_retries=3,
        verbose=True,
    ).run(**graph_inputs)
    _print_planning_result(result)

    if result.stage_output is None or not result.stage_output.approved:
        return

    save_choice = input("\nSave approved planning output to project store? [Y/n]: ").strip().lower()
    if save_choice in {"", "y", "yes"}:
        store_repository.save_planning(project_id, result.stage_output)
        print("Saved approved planning output to Project Store.")


def _run_structured_agent(agent: Any, agent_input: BaseModel) -> Any:
    return agent.invoke(
        {
            "messages": [
                HumanMessage(content=agent_input.model_dump_json(indent=2))
            ]
        }
    )


def _coerce_model(value: Any, model_cls: type[ModelT]) -> ModelT:
    if isinstance(value, model_cls):
        return value
    if isinstance(value, dict):
        structured_response = value.get("structured_response")
        if structured_response is not None:
            return _coerce_model(structured_response, model_cls)
        return model_cls.model_validate(value)
    if isinstance(value, str):
        return model_cls.model_validate_json(value)
    if isinstance(value, BaseMessage):
        return _coerce_model(value.content, model_cls)
    raise TypeError(f"Cannot convert {type(value).__name__} to {model_cls.__name__}")


def _is_format_error(error: Exception) -> bool:
    current: BaseException | None = error
    while current is not None:
        if current.__class__.__name__ in {
            "StructuredOutputValidationError",
            "MultipleStructuredOutputsError",
            "ValidationError",
            "ValueError",
        }:
            return True
        current = current.__cause__ or current.__context__
    return isinstance(error, ValidationError)


def _extract_invalid_output(error: Exception) -> Any:
    current: BaseException | None = error
    while current is not None:
        ai_message = getattr(current, "ai_message", None)
        if ai_message is not None:
            if hasattr(ai_message, "model_dump"):
                return ai_message.model_dump(mode="json")
            return str(ai_message)
        current = current.__cause__ or current.__context__
    return None


def _add_format_feedback(agent_input: ModelT, feedback: str) -> ModelT:
    if not feedback:
        return agent_input

    if hasattr(agent_input, "previous_planning_feedback"):
        previous = getattr(agent_input, "previous_planning_feedback", "")
        return agent_input.model_copy(
            update={
                "previous_planning_feedback": _append_feedback(
                    previous,
                    f"Structured output correction advice:\n{feedback}",
                )
            }
        )

    if hasattr(agent_input, "format_feedback"):
        previous = getattr(agent_input, "format_feedback", "")
        return agent_input.model_copy(
            update={
                "format_feedback": _append_feedback(
                    previous,
                    f"Structured output correction advice:\n{feedback}",
                )
            }
        )

    return agent_input


def _build_planning_context(module_designs: dict[str, ModuleDesignOutput]) -> PlanningContext:
    modules: list[PlanningModuleContext] = []
    dependency_edges: list[PlanningDependencyContext] = []
    seen_edges: set[tuple[str, str]] = set()

    for component_name, module_design in module_designs.items():
        modules_by_name = {module.name: module for module in module_design.modules if module.name}
        for module in module_design.modules:
            if not module.name:
                continue
            signature_ids = [
                _signature_id(module.name, signature.name)
                for signature in module.signatures
                if signature.name
            ]
            dependencies = [
                dependency
                for dependency in module.dependencies
                if dependency in modules_by_name
            ]
            modules.append(
                PlanningModuleContext(
                    component_name=module.component or module_design.component_name or component_name,
                    module_name=module.name,
                    dependencies=dependencies,
                    signature_ids=signature_ids,
                )
            )
            for dependency in dependencies:
                edge_key = (dependency, module.name)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    dependency_edges.append(
                        PlanningDependencyContext(
                            source=dependency,
                            target=module.name,
                            component_name=module.component or module_design.component_name or component_name,
                        )
                    )

        for edge in module_design.dependency_graph.edges:
            normalised_edge = _normalise_dependency_edge(edge, modules_by_name)
            if normalised_edge is None:
                continue
            edge_key = (normalised_edge.source, normalised_edge.target)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            dependency_edges.append(
                PlanningDependencyContext(
                    source=normalised_edge.source,
                    target=normalised_edge.target,
                    component_name=module_design.component_name or component_name,
                )
            )

    module_names = [module.module_name for module in modules]
    signature_ids = [
        signature_id
        for module in modules
        for signature_id in module.signature_ids
    ]
    return PlanningContext(
        module_names=module_names,
        modules=modules,
        signature_ids=signature_ids,
        dependency_edges=dependency_edges,
    )


def _normalise_dependency_edge(
    edge: GraphDependencyEdgeSpec,
    modules_by_name: dict[str, ModuleSpec],
) -> GraphDependencyEdgeSpec | None:
    if edge.source not in modules_by_name or edge.target not in modules_by_name:
        return None
    return edge


def _validate_planning_artifacts(
    implementation_plan: ImplementationPlan,
    test_plan: TestPlanGraph,
    planning_context: PlanningContext,
) -> list[str]:
    errors: list[str] = []
    expected_modules = set(planning_context.module_names)
    planned_modules = {
        module_name
        for file_plan in implementation_plan.files
        for module_name in file_plan.modules
    }
    unknown_planned_modules = sorted(planned_modules - expected_modules)
    missing_modules = sorted(expected_modules - planned_modules)
    if missing_modules:
        errors.append(f"Implementation plan is missing modules: {', '.join(missing_modules)}.")
    if unknown_planned_modules:
        errors.append(f"Implementation plan references unknown modules: {', '.join(unknown_planned_modules)}.")

    for module_name in sorted(expected_modules):
        unit_tests = [
            unit_test
            for file_plan in implementation_plan.files
            if module_name in file_plan.modules
            for unit_test in file_plan.unit_tests
        ]
        if not unit_tests:
            errors.append(f"Implementation file plan for {module_name} must include unit_tests strings.")

    test_node_ids = [node.node_id for node in test_plan.nodes]
    if len(test_node_ids) != len(set(test_node_ids)):
        errors.append("Test plan node ids must be unique.")

    step_ids = [step.step_id for step in implementation_plan.steps]
    if len(step_ids) != len(set(step_ids)):
        errors.append("Implementation step ids must be unique.")

    unit_nodes = [node for node in test_plan.nodes if node.kind == "unit"]
    covered_unit_modules = {
        module_name
        for node in unit_nodes
        for module_name in node.target_modules
    }
    for module in planning_context.modules:
        if module.signature_ids:
            covered_signatures = {
                signature_id
                for node in unit_nodes
                for signature_id in node.target_signatures
            }
            missing_signatures = sorted(set(module.signature_ids) - covered_signatures)
            if missing_signatures:
                errors.append(
                    f"Test plan is missing unit coverage for signatures: {', '.join(missing_signatures)}."
                )
        elif module.module_name not in covered_unit_modules:
            errors.append(f"Test plan is missing unit coverage for module {module.module_name}.")

    integration_nodes = [
        node
        for node in test_plan.nodes
        if node.kind in {"module_integration", "component_integration", "system_integration"}
    ]
    for dependency_edge in planning_context.dependency_edges:
        if not any(
            dependency_edge.source in node.target_modules
            and dependency_edge.target in node.target_modules
            for node in integration_nodes
        ):
            errors.append(
                "Test plan is missing integration coverage for dependency "
                f"{dependency_edge.source} -> {dependency_edge.target}."
            )

    system_nodes = [node for node in test_plan.nodes if node.kind == "system_integration"]
    if len(system_nodes) != 1:
        errors.append("Test plan must contain exactly one system_integration root node.")
    elif test_plan.root_node_id != system_nodes[0].node_id:
        errors.append("Test plan root_node_id must reference the single system_integration node.")

    cycle = _find_test_plan_cycle(test_plan)
    if cycle:
        errors.append(f"Test plan dependency graph contains a cycle: {' -> '.join(cycle)}.")

    serialised_output = json.dumps(
        {
            "implementation_plan": implementation_plan.model_dump(mode="json"),
            "test_plan": test_plan.model_dump(mode="json"),
        }
    )
    if "question_answer_context" in serialised_output:
        errors.append("Planning artifacts must not include question_answer_context.")

    return errors


def _find_test_plan_cycle(test_plan: TestPlanGraph) -> list[str]:
    adjacency: dict[str, list[str]] = {node.node_id: [] for node in test_plan.nodes}
    for node in test_plan.nodes:
        for dependency_id in node.depends_on:
            if dependency_id in adjacency:
                adjacency[dependency_id].append(node.node_id)
    for edge in test_plan.edges:
        if edge.source in adjacency and edge.target in adjacency:
            adjacency[edge.source].append(edge.target)

    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(node_id: str) -> list[str] | None:
        if node_id in visiting:
            start_index = stack.index(node_id)
            return [*stack[start_index:], node_id]
        if node_id in visited:
            return None
        visiting.add(node_id)
        stack.append(node_id)
        for next_id in adjacency.get(node_id, []):
            cycle = visit(next_id)
            if cycle:
                return cycle
        stack.pop()
        visiting.remove(node_id)
        visited.add(node_id)
        return None

    for node_id in adjacency:
        cycle = visit(node_id)
        if cycle:
            return cycle
    return []


def _signature_id(module_name: str, signature_name: str) -> str:
    return f"{module_name}.{signature_name}"


def _format_revision_feedback(
    validation_errors: list[str],
    critic_output: PlanningCriticOutput,
) -> str:
    feedback_parts = ["Previous planning artifacts failed review."]
    if validation_errors:
        feedback_parts.append("Validation errors:\n- " + "\n- ".join(validation_errors))
    if critic_output.feedback:
        feedback_parts.append(f"Critic feedback:\n{critic_output.feedback}")
    if critic_output.required_changes:
        feedback_parts.append("Required changes:\n- " + "\n- ".join(critic_output.required_changes))
    return "\n\n".join(feedback_parts)


def _append_feedback(existing_feedback: str, new_feedback: str) -> str:
    if not existing_feedback:
        return new_feedback
    if not new_feedback:
        return existing_feedback
    return f"{existing_feedback}\n\n{new_feedback}"


def _format_exception(error: Exception) -> str:
    message = str(error).strip()
    if not message:
        message = error.__class__.__name__
    return f"{error.__class__.__name__}: {message}"


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"\nError: {exc}") from None

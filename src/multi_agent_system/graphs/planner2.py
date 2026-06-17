from __future__ import annotations

from typing import Any, Callable, Literal, TypedDict, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.agents.AgentFactory import AgentFactory
from src.multi_agent_system.graphs.planning import (
    FormatErrorCallback,
    FormatErrorDecision,
    ImplementationPlannerInput,
    PlanningContext,
    PlanningCriticInput,
    PlanningCriticOutput,
    PlanningRunResult,
    _add_format_feedback,
    _append_feedback,
    _build_planning_context,
    _coerce_model,
    _extract_invalid_output,
    _format_exception,
    _format_revision_feedback,
    _interactive_format_error_decision,
    _is_format_error,
    _load_planning_inputs_from_store,
    _print_loaded_planning_inputs,
    _print_planning_result,
    _signature_id,
    _validate_planning_artifacts,
)
from src.multi_agent_system.output_schema import (
    ComponentDecompositionOutput,
    ComponentExtractionOutput,
    EnvironmentSetupPlan,
    ImplementationPlan,
    PlanningGraphInput,
    PlanningStageOutput,
    RepositoryStructure,
    RequirementsSpec,
    TestPlanEdge,
    TestPlanGraph,
    TestPlanNode,
)
from src.multi_agent_system.store import ProjectStoreRepository
from src.settings import configure_langsmith_environment, settings


Planner2Status = Literal["complete", "needs_revision", "failed"]
AgentRunner = Callable[[Any, BaseModel], Any]
ModelT = TypeVar("ModelT", bound=BaseModel)


IMPLEMENTATION_PLANNER_PROMPT = """
You are the Implementation Planner in Planner2.

Create an ImplementationPlan that maps every provided module to implementation files.
Keep file unit_tests as simple strings. Cover every module in planning_context.module_names.
Return only the ImplementationPlan artifact.
""".strip()


UNIT_TEST_PLANNER_PROMPT = """
You are the Unit Test Planner in Planner2.

Create only unit TestPlanNode objects. Your only job is signature-level unit coverage:
- For every signature in planning_context.signature_ids, include it in a unit node target_signatures list.
- If a module has no signatures, create a unit node that targets the module.
- Do not create integration or system nodes.
- Keep test_cases as simple strings.
""".strip()


DEPENDENCY_INTEGRATION_PLANNER_PROMPT = """
You are the Dependency Integration Planner in Planner2.

Create only module_integration TestPlanNode objects. Your only job is dependency coverage:
- For every dependency in planning_context.dependency_edges, create or reuse a module_integration node
  whose target_modules include both dependency modules.
- Do not create unit, component_integration, or system_integration nodes.
- Do not set depends_on. The deterministic assembler will wire dependencies safely.
""".strip()


PLANNING_CRITIC_PROMPT = """
You are the Planner2 Critic.

Review the ImplementationPlan and assembled TestPlanGraph for usefulness and consistency.
If validation_errors is non-empty, set approved=false and include those errors in required_changes.
Approve only when the artifacts are concrete enough for a later implementation graph.
""".strip()


class UnitTestPlan(BaseModel):
    nodes: list[TestPlanNode] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DependencyIntegrationPlan(BaseModel):
    nodes: list[TestPlanNode] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class UnitTestPlannerInput(BaseModel):
    project_prompt: str
    requirements: RequirementsSpec
    component_extraction: ComponentExtractionOutput
    component_decompositions: dict[str, ComponentDecompositionOutput] = Field(default_factory=dict)
    planning_context: PlanningContext
    implementation_plan: ImplementationPlan
    previous_planning_feedback: str = ""


class DependencyIntegrationPlannerInput(BaseModel):
    project_prompt: str
    requirements: RequirementsSpec
    component_extraction: ComponentExtractionOutput
    planning_context: PlanningContext
    unit_test_plan: UnitTestPlan
    previous_planning_feedback: str = ""


class Planner2State(TypedDict, total=False):
    graph_input: PlanningGraphInput
    planning_context: PlanningContext
    previous_planning_feedback: str
    implementation_plan: ImplementationPlan
    unit_test_plan: UnitTestPlan
    dependency_integration_plan: DependencyIntegrationPlan
    test_plan: TestPlanGraph
    validation_errors: list[str]
    critic_output: PlanningCriticOutput
    stage_output: PlanningStageOutput
    revision_count: int
    max_iterations: int
    status: Planner2Status
    failure_reason: str


class Planner2Graph:
    """Planner2 splits test planning into smaller agents and deterministic assembly."""

    def __init__(
        self,
        implementation_planner_agent: Any | None = None,
        unit_test_planner_agent: Any | None = None,
        dependency_integration_planner_agent: Any | None = None,
        critic_agent: Any | None = None,
        implementation_planner_runner: AgentRunner | None = None,
        unit_test_planner_runner: AgentRunner | None = None,
        dependency_integration_planner_runner: AgentRunner | None = None,
        critic_runner: AgentRunner | None = None,
        format_error_callback: FormatErrorCallback | None = None,
        max_format_retries: int = 2,
        verbose: bool = True,
        model_name: str = "gpt-5.3-codex",
        temperature: float = 0.2,
    ) -> None:
        self.implementation_planner_agent = implementation_planner_agent
        self.unit_test_planner_agent = unit_test_planner_agent
        self.dependency_integration_planner_agent = dependency_integration_planner_agent
        self.critic_agent = critic_agent
        self.implementation_planner_runner = implementation_planner_runner or _run_structured_agent
        self.unit_test_planner_runner = unit_test_planner_runner or _run_structured_agent
        self.dependency_integration_planner_runner = dependency_integration_planner_runner or _run_structured_agent
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
        module_designs: dict[str, Any],
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
        initial_state: Planner2State = {
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
        graph_builder = StateGraph(Planner2State)
        graph_builder.add_node("prepare_context", self.prepare_context)
        graph_builder.add_node("implementation_planner", self.implementation_planner)
        graph_builder.add_node("unit_test_planner", self.unit_test_planner)
        graph_builder.add_node("dependency_integration_planner", self.dependency_integration_planner)
        graph_builder.add_node("test_graph_assembler", self.test_graph_assembler)
        graph_builder.add_node("deterministic_validator", self.deterministic_validator)
        graph_builder.add_node("critic_review", self.critic_review)
        graph_builder.add_node("complete", self.complete)
        graph_builder.add_node("failed", self.failed)

        graph_builder.add_edge(START, "prepare_context")
        graph_builder.add_edge("prepare_context", "implementation_planner")
        graph_builder.add_edge("implementation_planner", "unit_test_planner")
        graph_builder.add_edge("unit_test_planner", "dependency_integration_planner")
        graph_builder.add_edge("dependency_integration_planner", "test_graph_assembler")
        graph_builder.add_edge("test_graph_assembler", "deterministic_validator")
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

    def prepare_context(self, state: Planner2State) -> Planner2State:
        self._announce_node("prepare_context")
        planning_context = _build_planning_context(state["graph_input"].module_designs)
        self._print_context_summary(planning_context)
        return {
            **state,
            "planning_context": planning_context,
        }

    def implementation_planner(self, state: Planner2State) -> Planner2State:
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
        implementation_plan = self._run_agent_with_format_retries(
            "implementation_planner",
            self._get_implementation_planner_agent(),
            planner_input,
            self.implementation_planner_runner,
            ImplementationPlan,
        )
        self._print_implementation_plan_summary(implementation_plan, state["planning_context"])
        return {
            **state,
            "implementation_plan": implementation_plan,
        }

    def unit_test_planner(self, state: Planner2State) -> Planner2State:
        self._announce_node("unit_test_planner")
        graph_input = state["graph_input"]
        planner_input = UnitTestPlannerInput(
            project_prompt=graph_input.project_prompt,
            requirements=graph_input.requirements,
            component_extraction=graph_input.component_extraction,
            component_decompositions=graph_input.component_decompositions,
            planning_context=state["planning_context"],
            implementation_plan=state["implementation_plan"],
            previous_planning_feedback=state.get("previous_planning_feedback", ""),
        )
        unit_test_plan = self._run_agent_with_format_retries(
            "unit_test_planner",
            self._get_unit_test_planner_agent(),
            planner_input,
            self.unit_test_planner_runner,
            UnitTestPlan,
        )
        self._print_unit_test_plan_summary(unit_test_plan, state["planning_context"])
        return {
            **state,
            "unit_test_plan": unit_test_plan,
        }

    def dependency_integration_planner(self, state: Planner2State) -> Planner2State:
        self._announce_node("dependency_integration_planner")
        graph_input = state["graph_input"]
        planner_input = DependencyIntegrationPlannerInput(
            project_prompt=graph_input.project_prompt,
            requirements=graph_input.requirements,
            component_extraction=graph_input.component_extraction,
            planning_context=state["planning_context"],
            unit_test_plan=state["unit_test_plan"],
            previous_planning_feedback=state.get("previous_planning_feedback", ""),
        )
        dependency_integration_plan = self._run_agent_with_format_retries(
            "dependency_integration_planner",
            self._get_dependency_integration_planner_agent(),
            planner_input,
            self.dependency_integration_planner_runner,
            DependencyIntegrationPlan,
        )
        self._print_dependency_plan_summary(dependency_integration_plan, state["planning_context"])
        return {
            **state,
            "dependency_integration_plan": dependency_integration_plan,
        }

    def test_graph_assembler(self, state: Planner2State) -> Planner2State:
        self._announce_node("test_graph_assembler")
        test_plan = _assemble_test_plan_graph(
            state["unit_test_plan"],
            state["dependency_integration_plan"],
            state["planning_context"],
        )
        self._print_test_plan_summary(test_plan)
        return {
            **state,
            "test_plan": test_plan,
        }

    def deterministic_validator(self, state: Planner2State) -> Planner2State:
        self._announce_node("deterministic_validator")
        validation_errors = _validate_planning_artifacts(
            state["implementation_plan"],
            state["test_plan"],
            state["planning_context"],
        )
        self._print_validation_summary(validation_errors)
        return {
            **state,
            "validation_errors": validation_errors,
        }

    def critic_review(self, state: Planner2State) -> Planner2State:
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
            "critic_review",
            self._get_critic_agent(),
            critic_input,
            self.critic_runner,
            PlanningCriticOutput,
        )
        self._print_critic_summary(critic_output, state.get("validation_errors", []))

        if state.get("validation_errors") or not critic_output.approved:
            revision_count = state.get("revision_count", 0) + 1
            next_feedback = _append_feedback(
                state.get("previous_planning_feedback", ""),
                _format_revision_feedback(state.get("validation_errors", []), critic_output),
            )
            next_state: Planner2State = {
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
                        "Planner2 artifacts did not pass review before the iteration limit "
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
    def route_after_critic(state: Planner2State) -> str:
        if state.get("status") == "complete":
            print("Planner2 route: critic_review -> complete")
            return "complete"
        if state.get("status") == "failed":
            print("Planner2 route: critic_review -> failed")
            return "failed"
        print("Planner2 route: critic_review -> implementation_planner")
        return "implementation_planner"

    @staticmethod
    def complete(state: Planner2State) -> Planner2State:
        print("Planner2 node: complete")
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
    def failed(state: Planner2State) -> Planner2State:
        print("Planner2 node: failed")
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
            "failure_reason": state.get("failure_reason", "Planner2 failed."),
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

    def _get_unit_test_planner_agent(self) -> Any:
        if self.unit_test_planner_agent is None:
            self.unit_test_planner_agent = AgentFactory.build_agent(
                prompt=UNIT_TEST_PLANNER_PROMPT,
                tools=[],
                temperature=self.temperature,
                model_name=self.model_name,
                response_format=UnitTestPlan,
            )
        return self.unit_test_planner_agent

    def _get_dependency_integration_planner_agent(self) -> Any:
        if self.dependency_integration_planner_agent is None:
            self.dependency_integration_planner_agent = AgentFactory.build_agent(
                prompt=DEPENDENCY_INTEGRATION_PLANNER_PROMPT,
                tools=[],
                temperature=self.temperature,
                model_name=self.model_name,
                response_format=DependencyIntegrationPlan,
            )
        return self.dependency_integration_planner_agent

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
            print(f"Planner2 node: {node_name}")

    def _print_context_summary(self, planning_context: PlanningContext) -> None:
        if not self.verbose:
            return
        print(
            "Planner2 context: "
            f"modules={len(planning_context.module_names)}, "
            f"signatures={len(planning_context.signature_ids)}, "
            f"dependencies={len(planning_context.dependency_edges)}"
        )
        _print_limited_items("Expected modules", planning_context.module_names)

    def _print_implementation_plan_summary(
        self,
        implementation_plan: ImplementationPlan,
        planning_context: PlanningContext,
    ) -> None:
        if not self.verbose:
            return
        planned_modules = [
            module_name
            for file_plan in implementation_plan.files
            for module_name in file_plan.modules
        ]
        missing_modules = sorted(set(planning_context.module_names) - set(planned_modules))
        files_without_tests = [
            file_plan.relative_path
            for file_plan in implementation_plan.files
            if not file_plan.unit_tests
        ]
        print(
            "Planner2 implementation plan: "
            f"files={len(implementation_plan.files)}, "
            f"covered_modules={len(set(planned_modules))}/{len(set(planning_context.module_names))}, "
            f"steps={len(implementation_plan.steps)}"
        )
        _print_limited_items("Implementation files", [
            f"{file_plan.relative_path} -> {', '.join(file_plan.modules) or 'no modules'}"
            for file_plan in implementation_plan.files
        ])
        if missing_modules:
            _print_limited_items("Missing implementation modules", missing_modules)
        if files_without_tests:
            _print_limited_items("Files without unit_tests", files_without_tests)

    def _print_unit_test_plan_summary(
        self,
        unit_test_plan: UnitTestPlan,
        planning_context: PlanningContext,
    ) -> None:
        if not self.verbose:
            return
        covered_modules = {
            module_name
            for node in unit_test_plan.nodes
            for module_name in node.target_modules
        }
        covered_signatures = {
            signature_id
            for node in unit_test_plan.nodes
            for signature_id in node.target_signatures
        }
        missing_modules = sorted(set(planning_context.module_names) - covered_modules)
        missing_signatures = sorted(set(planning_context.signature_ids) - covered_signatures)
        print(
            "Planner2 unit test plan: "
            f"nodes={len(unit_test_plan.nodes)}, "
            f"covered_modules={len(covered_modules)}/{len(set(planning_context.module_names))}, "
            f"covered_signatures={len(covered_signatures)}/{len(set(planning_context.signature_ids))}"
        )
        if missing_modules:
            _print_limited_items("Unit plan missing modules", missing_modules)
        if missing_signatures:
            _print_limited_items("Unit plan missing signatures", missing_signatures)

    def _print_dependency_plan_summary(
        self,
        dependency_plan: DependencyIntegrationPlan,
        planning_context: PlanningContext,
    ) -> None:
        if not self.verbose:
            return
        print(
            "Planner2 dependency integration plan: "
            f"nodes={len(dependency_plan.nodes)}, "
            f"expected_dependencies={len(planning_context.dependency_edges)}"
        )
        _print_limited_items("Dependency integration nodes", [
            f"{node.node_id}: {', '.join(node.target_modules) or 'no modules'}"
            for node in dependency_plan.nodes
        ])

    def _print_test_plan_summary(self, test_plan: TestPlanGraph) -> None:
        if not self.verbose:
            return
        kind_counts: dict[str, int] = {}
        for node in test_plan.nodes:
            kind_counts[node.kind] = kind_counts.get(node.kind, 0) + 1
        kind_text = ", ".join(f"{kind}={count}" for kind, count in sorted(kind_counts.items())) or "none"
        print(
            "Planner2 assembled test graph: "
            f"nodes={len(test_plan.nodes)}, edges={len(test_plan.edges)}, "
            f"root={test_plan.root_node_id or 'None'}, kinds={kind_text}"
        )

    def _print_validation_summary(self, validation_errors: list[str]) -> None:
        if not self.verbose:
            return
        if not validation_errors:
            print("Planner2 validation: passed")
            return
        print(f"Planner2 validation: failed with {len(validation_errors)} issue(s)")
        _print_limited_items("Validation errors", validation_errors)

    def _print_critic_summary(
        self,
        critic_output: PlanningCriticOutput,
        validation_errors: list[str],
    ) -> None:
        if not self.verbose:
            return
        print(
            "Planner2 critic: "
            f"approved={critic_output.approved}, "
            f"required_changes={len(critic_output.required_changes)}, "
            f"validation_errors={len(validation_errors)}"
        )
        if critic_output.verdict:
            print(f"Critic verdict: {_truncate(critic_output.verdict, 500)}")
        if critic_output.required_changes:
            _print_limited_items("Critic required changes", critic_output.required_changes)

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
                    node_name,
                    current_input,
                    error,
                    attempt,
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
        if self.format_error_callback is not None:
            return self.format_error_callback(
                node_name,
                agent_input,
                error,
                attempt,
                self.max_format_retries,
            )
        return FormatErrorDecision(retry=False)

    @staticmethod
    def _to_result(state: Planner2State) -> PlanningRunResult:
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


def build_planner2_graph(**kwargs: Any) -> Planner2Graph:
    return Planner2Graph(**kwargs)


def _print_limited_items(label: str, items: list[str], limit: int = 12) -> None:
    if not items:
        return
    print(f"{label}:")
    for item in items[:limit]:
        print(f"- {_truncate(item, 240)}")
    remaining = len(items) - limit
    if remaining > 0:
        print(f"- ... {remaining} more")


def _truncate(value: str, max_length: int) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3]}..."


def _assemble_test_plan_graph(
    unit_test_plan: UnitTestPlan,
    dependency_plan: DependencyIntegrationPlan,
    planning_context: PlanningContext,
) -> TestPlanGraph:
    nodes_by_id: dict[str, TestPlanNode] = {}
    unit_node_ids_by_module: dict[str, list[str]] = {}
    integration_node_ids: list[str] = []
    edges: list[TestPlanEdge] = []

    for node in unit_test_plan.nodes:
        if node.kind != "unit" or not node.node_id:
            continue
        clean_node = node.model_copy(update={"depends_on": []})
        nodes_by_id[clean_node.node_id] = clean_node
        for module_name in clean_node.target_modules:
            unit_node_ids_by_module.setdefault(module_name, []).append(clean_node.node_id)

    _ensure_unit_coverage(nodes_by_id, unit_node_ids_by_module, planning_context)

    for node in dependency_plan.nodes:
        if node.kind != "module_integration" or len(node.target_modules) < 2:
            continue
        node_id = node.node_id or _dependency_node_id(node.target_modules[0], node.target_modules[1])
        depends_on = _unit_dependencies_for_modules(unit_node_ids_by_module, node.target_modules)
        clean_node = node.model_copy(update={"node_id": node_id, "depends_on": depends_on})
        nodes_by_id[node_id] = clean_node
        integration_node_ids.append(node_id)

    _ensure_dependency_integration_coverage(
        nodes_by_id,
        unit_node_ids_by_module,
        integration_node_ids,
        planning_context,
    )

    component_node_ids = _add_component_integration_nodes(
        nodes_by_id,
        unit_node_ids_by_module,
        integration_node_ids,
        planning_context,
    )
    root_node_id = "system_integration"
    root_dependencies = sorted(set(component_node_ids or integration_node_ids or _all_unit_node_ids(unit_node_ids_by_module)))
    nodes_by_id[root_node_id] = TestPlanNode(
        node_id=root_node_id,
        kind="system_integration",
        title="Whole system integration",
        target_modules=planning_context.module_names,
        test_cases=["Validate the complete system workflow across all planned modules."],
        depends_on=root_dependencies,
    )

    for node in nodes_by_id.values():
        for dependency_id in node.depends_on:
            if dependency_id in nodes_by_id:
                edges.append(TestPlanEdge(source=dependency_id, target=node.node_id))

    return TestPlanGraph(
        testing_framework="pytest",
        nodes=list(nodes_by_id.values()),
        edges=_dedupe_edges(edges),
        root_node_id=root_node_id,
        commands=["pytest"],
    )


def _ensure_unit_coverage(
    nodes_by_id: dict[str, TestPlanNode],
    unit_node_ids_by_module: dict[str, list[str]],
    planning_context: PlanningContext,
) -> None:
    covered_signatures = {
        signature_id
        for node in nodes_by_id.values()
        if node.kind == "unit"
        for signature_id in node.target_signatures
    }
    for module in planning_context.modules:
        missing_signatures = [signature_id for signature_id in module.signature_ids if signature_id not in covered_signatures]
        if not module.signature_ids and module.module_name not in unit_node_ids_by_module:
            node_id = f"unit_{_slug(module.module_name)}"
            nodes_by_id[node_id] = TestPlanNode(
                node_id=node_id,
                kind="unit",
                title=f"{module.module_name} unit tests",
                target_modules=[module.module_name],
                test_cases=[f"{module.module_name} supports its module responsibilities."],
            )
            unit_node_ids_by_module.setdefault(module.module_name, []).append(node_id)
            continue
        if not missing_signatures:
            continue
        node_id = f"unit_{_slug(module.module_name)}"
        existing = nodes_by_id.get(node_id)
        if existing is None:
            nodes_by_id[node_id] = TestPlanNode(
                node_id=node_id,
                kind="unit",
                title=f"{module.module_name} unit tests",
                target_modules=[module.module_name],
                target_signatures=missing_signatures,
                test_cases=[
                    f"{signature_id} behaves according to its signature contract."
                    for signature_id in missing_signatures
                ],
            )
            unit_node_ids_by_module.setdefault(module.module_name, []).append(node_id)
            continue
        merged_signatures = sorted(set(existing.target_signatures) | set(missing_signatures))
        nodes_by_id[node_id] = existing.model_copy(
            update={
                "target_modules": sorted(set(existing.target_modules) | {module.module_name}),
                "target_signatures": merged_signatures,
                "test_cases": [
                    *existing.test_cases,
                    *[
                        f"{signature_id} behaves according to its signature contract."
                        for signature_id in missing_signatures
                    ],
                ],
            }
        )
        unit_node_ids_by_module.setdefault(module.module_name, []).append(node_id)


def _ensure_dependency_integration_coverage(
    nodes_by_id: dict[str, TestPlanNode],
    unit_node_ids_by_module: dict[str, list[str]],
    integration_node_ids: list[str],
    planning_context: PlanningContext,
) -> None:
    for dependency in planning_context.dependency_edges:
        if any(
            dependency.source in nodes_by_id[node_id].target_modules
            and dependency.target in nodes_by_id[node_id].target_modules
            for node_id in integration_node_ids
            if node_id in nodes_by_id
        ):
            continue
        node_id = _dependency_node_id(dependency.source, dependency.target)
        nodes_by_id[node_id] = TestPlanNode(
            node_id=node_id,
            kind="module_integration",
            title=f"{dependency.source} and {dependency.target} integration",
            target_modules=[dependency.source, dependency.target],
            test_cases=[
                f"{dependency.target} integrates correctly with {dependency.source}.",
            ],
            depends_on=_unit_dependencies_for_modules(
                unit_node_ids_by_module,
                [dependency.source, dependency.target],
            ),
        )
        integration_node_ids.append(node_id)


def _add_component_integration_nodes(
    nodes_by_id: dict[str, TestPlanNode],
    unit_node_ids_by_module: dict[str, list[str]],
    integration_node_ids: list[str],
    planning_context: PlanningContext,
) -> list[str]:
    modules_by_component: dict[str, list[str]] = {}
    for module in planning_context.modules:
        modules_by_component.setdefault(module.component_name or "application", []).append(module.module_name)

    component_node_ids: list[str] = []
    for component_name, module_names in modules_by_component.items():
        node_id = f"component_{_slug(component_name)}"
        dependency_ids = [
            integration_id
            for integration_id in integration_node_ids
            if any(module_name in nodes_by_id[integration_id].target_modules for module_name in module_names)
        ]
        if not dependency_ids:
            dependency_ids = _unit_dependencies_for_modules(unit_node_ids_by_module, module_names)
        nodes_by_id[node_id] = TestPlanNode(
            node_id=node_id,
            kind="component_integration",
            title=f"{component_name} component integration",
            target_modules=module_names,
            test_cases=[f"{component_name} modules work together through the component workflow."],
            depends_on=sorted(set(dependency_ids)),
        )
        component_node_ids.append(node_id)
    return component_node_ids


def _unit_dependencies_for_modules(
    unit_node_ids_by_module: dict[str, list[str]],
    module_names: list[str],
) -> list[str]:
    return sorted(
        {
            node_id
            for module_name in module_names
            for node_id in unit_node_ids_by_module.get(module_name, [])
        }
    )


def _all_unit_node_ids(unit_node_ids_by_module: dict[str, list[str]]) -> list[str]:
    return sorted({node_id for node_ids in unit_node_ids_by_module.values() for node_id in node_ids})


def _dedupe_edges(edges: list[TestPlanEdge]) -> list[TestPlanEdge]:
    seen: set[tuple[str, str]] = set()
    deduped: list[TestPlanEdge] = []
    for edge in edges:
        edge_key = (edge.source, edge.target)
        if edge_key in seen:
            continue
        seen.add(edge_key)
        deduped.append(edge)
    return deduped


def _dependency_node_id(source: str, target: str) -> str:
    return f"integration_{_slug(source)}_{_slug(target)}"


def _slug(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif cleaned and cleaned[-1] != "_":
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "node"


def _run_structured_agent(agent: Any, agent_input: BaseModel) -> Any:
    return agent.invoke(
        {
            "messages": [
                HumanMessage(content=agent_input.model_dump_json(indent=2))
            ]
        }
    )


def main() -> None:
    project_id = input("Project id to run Planner2 [todo_cli]: ").strip() or "todo_cli"
    store_repository = ProjectStoreRepository()
    inputs = _load_planning_inputs_from_store(project_id, store_repository)
    _print_loaded_planning_inputs(inputs)

    choice = input("\nRun Planner2 with these inputs? [y/N]: ").strip().lower()
    if choice not in {"y", "yes"}:
        return

    graph_inputs = {key: value for key, value in inputs.items() if key != "project_id"}
    result = build_planner2_graph(
        format_error_callback=_interactive_format_error_decision,
        max_format_retries=3,
        verbose=True,
    ).run(**graph_inputs)
    _print_planning_result(result)

    if result.stage_output is None or not result.stage_output.approved:
        return

    save_choice = input("\nSave approved Planner2 output to project store? [Y/n]: ").strip().lower()
    if save_choice in {"", "y", "yes"}:
        store_repository.save_planning(project_id, result.stage_output)
        print("Saved approved Planner2 output to Project Store.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"\nError: {exc}") from None

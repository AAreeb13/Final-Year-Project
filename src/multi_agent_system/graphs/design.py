from __future__ import annotations

import json
from typing import Any, Callable, Literal, TypedDict, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.agents.AgentFactory import AgentFactory
from src.multi_agent_system.output_schema import (
    ArchitectInput,
    ArchitectOutput,
    ComponentSpec,
    DesignCriticInput,
    DesignCriticOutput,
    DesignStageOutput,
    DesignTask,
    QuestionAnswer,
    RequirementsStageOutput,
)


DesignStatus = Literal["complete", "needs_revision", "failed"]
AgentRunner = Callable[[Any, BaseModel], Any]
ModelT = TypeVar("ModelT", bound=BaseModel)


class UserApprovalDecision(BaseModel):
    approved: bool = True
    feedback: str = ""


ApprovalCallback = Callable[[str, dict[str, Any], dict[str, Any]], UserApprovalDecision]


def auto_approve(
    node_name: str,
    step_input: dict[str, Any],
    step_output: dict[str, Any],
) -> UserApprovalDecision:
    return UserApprovalDecision(approved=True)


ARCHITECT_PROMPT = """
You are the Architect Agent in a multi-agent SDLC system.

Use the DesignTask to determine your current job:
- extract_components: identify high-level architecture, components, relationships, and technologies.
- decompose_component: decompose the target component into modules.
- design_modules: design the target component's modules, signatures, dependencies, and dependency graph.

Use only the approved requirements, previous design output, question-answer context, and critic feedback provided in the input.
Return exactly one matching artifact inside ArchitectOutput.
""".strip()


DESIGN_CRITIC_PROMPT = """
You are Critic B for the design phase of a multi-agent SDLC system.

Review the ArchitectOutput according to the DesignTask:
- extract_components: check requirement coverage, separation of concerns, relationships, and technology choices.
- decompose_component: check responsibility clarity, inputs/outputs, coupling, and module boundaries.
- design_modules: check module purpose, granularity, testability, dependency direction, SOLID issues, and signatures.

Do not ask the user questions during the design phase.
If details are missing, make a conservative design decision or ask the Architect to revise through required_changes and feedback.
If architecture/design changes are needed, set approved=false and return required_changes and feedback.
Approve only when the current design artifact is good enough for the next SDLC step.
""".strip()


class DesignState(TypedDict, total=False):
    project_prompt: str
    requirements: RequirementsStageOutput
    task: DesignTask
    previous_design_output: DesignStageOutput | None
    question_answer_context: list[QuestionAnswer]
    critic_feedback: str
    architect_output: ArchitectOutput
    critic_output: DesignCriticOutput
    stage_output: DesignStageOutput
    status: DesignStatus
    approval_feedback: str
    approval_rejected_at: str


class DesignRunResult(BaseModel):
    status: DesignStatus
    task: DesignTask
    architect_output: ArchitectOutput | None = None
    critic_output: DesignCriticOutput | None = None
    stage_output: DesignStageOutput | None = None
    question_answer_context: list[QuestionAnswer] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)

    @property
    def components(self) -> list[ComponentSpec]:
        if not self.stage_output or not self.stage_output.component_extraction:
            return []
        return self.stage_output.component_extraction.components


class DesignStateGraph:
    """Architect + Critic graph for one design sub-task at a time."""

    def __init__(
        self,
        architect_agent: Any | None = None,
        critic_agent: Any | None = None,
        architect_runner: AgentRunner | None = None,
        critic_runner: AgentRunner | None = None,
        approval_callback: ApprovalCallback | None = None,
        auto_approval: bool = False,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ) -> None:
        self.architect_agent = architect_agent
        self.critic_agent = critic_agent
        self.architect_runner = architect_runner or _run_structured_agent
        self.critic_runner = critic_runner or _run_structured_agent
        self.auto_approval = auto_approval
        self.approval_callback = auto_approve if auto_approval or approval_callback is None else approval_callback
        self.model_name = model_name
        self.temperature = temperature
        self.graph = self._build_design_graph()

    def run(
        self,
        project_prompt: str,
        requirements: RequirementsStageOutput,
        task: DesignTask,
        previous_design_output: DesignStageOutput | None = None,
        question_answer_context: list[QuestionAnswer] | None = None,
        critic_feedback: str = "",
    ) -> DesignRunResult:
        initial_state: DesignState = {
            "project_prompt": project_prompt,
            "requirements": requirements,
            "task": task,
            "previous_design_output": previous_design_output,
            "question_answer_context": question_answer_context or [],
            "critic_feedback": critic_feedback,
        }
        final_state = self.graph.invoke(initial_state)
        return self._to_result(final_state)

    def _build_design_graph(self):
        graph_builder = StateGraph(DesignState)
        graph_builder.add_node("architect_design", self.architect_design)
        graph_builder.add_node("critic_review", self.critic_review)
        graph_builder.add_node("complete", self.complete)
        graph_builder.add_node("needs_revision", self.needs_revision)

        graph_builder.add_edge(START, "architect_design")
        graph_builder.add_conditional_edges(
            "architect_design",
            self.route_after_architect,
            {
                "critic_review": "critic_review",
                "needs_revision": "needs_revision",
            },
        )
        graph_builder.add_conditional_edges(
            "critic_review",
            self.route_after_critic,
            {
                "complete": "complete",
                "needs_revision": "needs_revision",
            },
        )
        graph_builder.add_edge("complete", END)
        graph_builder.add_edge("needs_revision", END)
        return graph_builder.compile()

    def architect_design(self, state: DesignState) -> DesignState:
        architect_input = ArchitectInput(
            project_prompt=state["project_prompt"],
            requirements=state["requirements"],
            task=state["task"],
            previous_design_output=state.get("previous_design_output"),
            question_answer_context=state.get("question_answer_context", []),
            critic_feedback=state.get("critic_feedback", ""),
        )
        raw_output = self.architect_runner(
            self._get_architect_agent(),
            architect_input,
        )
        architect_output = _coerce_model(raw_output, ArchitectOutput)
        architect_output = _normalise_architect_output_for_task(
            architect_output,
            state["task"],
        )
        approval = self.approval_callback(
            "architect_design",
            architect_input.model_dump(mode="json"),
            architect_output.model_dump(mode="json"),
        )
        if not approval.approved:
            critic_output = DesignCriticOutput(
                approved=False,
                verdict="Architect output rejected by user approval.",
                feedback=approval.feedback,
                required_changes=[approval.feedback] if approval.feedback else [],
            )
            return {
                **state,
                "architect_output": architect_output,
                "critic_output": critic_output,
                "approval_feedback": approval.feedback,
                "approval_rejected_at": "architect_design",
            }
        return {
            **state,
            "architect_output": architect_output,
        }

    def critic_review(self, state: DesignState) -> DesignState:
        critic_input = DesignCriticInput(
            project_prompt=state["project_prompt"],
            requirements=state["requirements"],
            task=state["task"],
            architect_output=state["architect_output"],
            question_answer_context=state.get("question_answer_context", []),
        )
        raw_output = self.critic_runner(
            self._get_critic_agent(),
            critic_input,
        )
        critic_output = _coerce_model(raw_output, DesignCriticOutput)
        approval = self.approval_callback(
            "critic_review",
            critic_input.model_dump(mode="json"),
            critic_output.model_dump(mode="json"),
        )
        if not approval.approved:
            feedback = approval.feedback or "Critic output rejected by user approval."
            critic_output.approved = False
            critic_output.feedback = _append_feedback(critic_output.feedback, feedback)
            if feedback not in critic_output.required_changes:
                critic_output.required_changes.append(feedback)
            return {
                **state,
                "critic_output": critic_output,
                "approval_feedback": approval.feedback,
                "approval_rejected_at": "critic_review",
            }
        return {
            **state,
            "critic_output": critic_output,
        }

    @staticmethod
    def route_after_architect(state: DesignState) -> str:
        if state.get("approval_rejected_at") == "architect_design":
            return "needs_revision"
        return "critic_review"

    @staticmethod
    def route_after_critic(state: DesignState) -> str:
        critic_output = state["critic_output"]
        if critic_output.questions:
            return "needs_revision"
        if critic_output.approved:
            return "complete"
        return "needs_revision"

    @staticmethod
    def complete(state: DesignState) -> DesignState:
        critic_output = state["critic_output"]
        stage_output = DesignStageOutput(
            task=state["task"],
            architect_output=state["architect_output"],
            question_answer_context=state.get("question_answer_context", []),
            critic_verdict=critic_output.verdict,
            approved=True,
        )
        return {
            **state,
            "stage_output": stage_output,
            "status": "complete",
        }

    @staticmethod
    def needs_revision(state: DesignState) -> DesignState:
        critic_output = state["critic_output"]
        stage_output = DesignStageOutput(
            task=state["task"],
            architect_output=state["architect_output"],
            question_answer_context=state.get("question_answer_context", []),
            critic_verdict=critic_output.verdict,
            approved=False,
        )
        return {
            **state,
            "stage_output": stage_output,
            "status": "needs_revision",
        }

    def _get_architect_agent(self) -> Any:
        if self.architect_agent is None:
            self.architect_agent = AgentFactory.build_agent(
                prompt=ARCHITECT_PROMPT,
                tools=[],
                temperature=self.temperature,
                model_name=self.model_name,
                response_format=ArchitectOutput,
            )
        return self.architect_agent

    def _get_critic_agent(self) -> Any:
        if self.critic_agent is None:
            self.critic_agent = AgentFactory.build_agent(
                prompt=DESIGN_CRITIC_PROMPT,
                tools=[],
                temperature=self.temperature,
                model_name=self.model_name,
                response_format=DesignCriticOutput,
            )
        return self.critic_agent

    @staticmethod
    def _to_result(state: DesignState) -> DesignRunResult:
        critic_output = state.get("critic_output")
        return DesignRunResult(
            status=state.get("status", "failed"),
            task=state["task"],
            architect_output=state.get("architect_output"),
            critic_output=critic_output,
            stage_output=state.get("stage_output"),
            question_answer_context=state.get("question_answer_context", []),
            questions=critic_output.questions if critic_output else [],
        )


def build_design_graph(**kwargs: Any) -> DesignStateGraph:
    return DesignStateGraph(**kwargs)


def _append_feedback(existing_feedback: str, new_feedback: str) -> str:
    if not existing_feedback:
        return new_feedback
    if not new_feedback:
        return existing_feedback
    return f"{existing_feedback}\n\nUser approval feedback: {new_feedback}"


def _normalise_architect_output_for_task(
    architect_output: ArchitectOutput,
    task: DesignTask,
) -> ArchitectOutput:
    if architect_output.task is None:
        architect_output = architect_output.model_copy(update={"task": task})
    if architect_output.task != task:
        raise ValueError(
            f"ArchitectOutput task {architect_output.task} does not match graph task {task}"
        )

    expected_artifacts = {
        "extract_components": architect_output.component_extraction,
        "decompose_component": architect_output.component_decomposition,
        "design_modules": architect_output.module_design,
    }
    if expected_artifacts[task.kind] is None:
        raise ValueError(f"{task.kind} output must populate its matching artifact field")
    return architect_output


def _run_structured_agent(agent: Any, agent_input: BaseModel) -> Any:
    payload = agent_input.model_dump(mode="json")
    return agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=json.dumps(payload, indent=2),
                )
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

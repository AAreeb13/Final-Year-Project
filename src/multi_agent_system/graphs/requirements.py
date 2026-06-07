from __future__ import annotations

import json
from typing import Any, Callable, Literal, TypedDict, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.agents.AgentFactory import AgentFactory
from src.multi_agent_system.output_schema import (
    QuestionAnswer,
    RequirementsCriticInput,
    RequirementsCriticOutput,
    RequirementsExtractorInput,
    RequirementsExtractorOutput,
    RequirementsSpec,
    RequirementsStageOutput,
)


RequirementsStatus = Literal["complete", "waiting_for_user"]
AgentRunner = Callable[[Any, BaseModel], Any]
ModelT = TypeVar("ModelT", bound=BaseModel)


EXTRACTOR_PROMPT = """
You are the Requirements Extractor Agent for a software SDLC multi-agent system.

Extract a clean requirements artifact from the project prompt. If previous
requirements and question-answer context are provided, update the requirements
using that context instead of duplicating it in notes.

Return the structured RequirementsExtractorOutput only.
""".strip()


CRITIC_PROMPT = """
You are Critic A for the requirements stage of a software SDLC multi-agent system.

Review the extracted requirements against the original project prompt and any
question-answer context.

Approve only when:
- functional and non-functional requirements are explicit enough to design from
- assumptions have either been removed or turned into answered context
- constraints are concrete
- out-of-scope items are clear

If you are not sure, set approved=false and return concise user-facing questions.
Return the structured RequirementsCriticOutput only.
""".strip()


class RequirementsState(TypedDict, total=False):
    project_prompt: str
    previous_requirements: RequirementsSpec | None
    question_answer_context: list[QuestionAnswer]
    extractor_output: RequirementsExtractorOutput
    critic_output: RequirementsCriticOutput
    stage_output: RequirementsStageOutput
    status: RequirementsStatus


class RequirementsRunResult(BaseModel):
    status: RequirementsStatus
    requirements: RequirementsSpec | None = None
    question_answer_context: list[QuestionAnswer] = Field(default_factory=list)
    critic_output: RequirementsCriticOutput | None = None
    stage_output: RequirementsStageOutput | None = None
    questions: list[str] = Field(default_factory=list)


class RequirementsStageGraph:
    """Small LangGraph for requirements extraction and Critic A review."""

    def __init__(
        self,
        extractor_agent: Any | None = None,
        critic_agent: Any | None = None,
        extractor_runner: AgentRunner | None = None,
        critic_runner: AgentRunner | None = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ) -> None:
        self.extractor_agent = extractor_agent
        self.critic_agent = critic_agent
        self.extractor_runner = extractor_runner or _run_structured_agent
        self.critic_runner = critic_runner or _run_structured_agent
        self.model_name = model_name
        self.temperature = temperature
        self.graph = self._build_requirement_graph()

    def run(
        self,
        project_prompt: str,
        previous_requirements: RequirementsSpec | None = None,
        question_answer_context: list[QuestionAnswer] | None = None,
    ) -> RequirementsRunResult:
        initial_state: RequirementsState = {
            "project_prompt": project_prompt,
            "previous_requirements": previous_requirements,
            "question_answer_context": question_answer_context or [],
        }
        final_state = self.graph.invoke(initial_state)
        return self._to_result(final_state)

    def _build_requirement_graph(self):
        graph_builder = StateGraph(RequirementsState)
        graph_builder.add_node("extract_requirements", self.extract_requirements)
        graph_builder.add_node("critic_review", self.critic_review)
        graph_builder.add_node("complete", self.complete)
        graph_builder.add_node("waiting_for_user", self.waiting_for_user)

        graph_builder.add_edge(START, "extract_requirements")
        graph_builder.add_edge("extract_requirements", "critic_review")
        graph_builder.add_conditional_edges(
            "critic_review",
            self.route_after_critic,
            {
                "complete": "complete",
                "waiting_for_user": "waiting_for_user",
            },
        )
        graph_builder.add_edge("complete", END)
        graph_builder.add_edge("waiting_for_user", END)
        graph = graph_builder.compile()
        print(graph)
        return graph

    def extract_requirements(self, state: RequirementsState) -> RequirementsState:
        extractor_input = RequirementsExtractorInput(
            project_prompt=state["project_prompt"],
            previous_requirements=state.get("previous_requirements"),
            question_answer_context=state.get("question_answer_context", []),
        )
        raw_output = self.extractor_runner(
            self._get_extractor_agent(),
            extractor_input,
        )
        extractor_output = _coerce_model(raw_output, RequirementsExtractorOutput)
        return {
            **state,
            "extractor_output": extractor_output,
            "previous_requirements": extractor_output.requirements,
        }

    def critic_review(self, state: RequirementsState) -> RequirementsState:
        extractor_output = state["extractor_output"]
        critic_input = RequirementsCriticInput(
            project_prompt=state["project_prompt"],
            requirements=extractor_output.requirements,
            question_answer_context=state.get("question_answer_context", []),
        )
        raw_output = self.critic_runner(
            self._get_critic_agent(),
            critic_input,
        )
        critic_output = _coerce_model(raw_output, RequirementsCriticOutput)
        return {
            **state,
            "critic_output": critic_output,
        }

    @staticmethod
    def route_after_critic(state: RequirementsState) -> str:
        critic_output = state["critic_output"]
        if critic_output.approved:
            return "complete"
        return "waiting_for_user"

    @staticmethod
    def complete(state: RequirementsState) -> RequirementsState:
        extractor_output = state["extractor_output"]
        critic_output = state["critic_output"]
        stage_output = RequirementsStageOutput(
            requirements=extractor_output.requirements,
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
    def waiting_for_user(state: RequirementsState) -> RequirementsState:
        extractor_output = state["extractor_output"]
        critic_output = state["critic_output"]
        stage_output = RequirementsStageOutput(
            requirements=extractor_output.requirements,
            question_answer_context=state.get("question_answer_context", []),
            critic_verdict=critic_output.verdict,
            approved=False,
        )
        return {
            **state,
            "stage_output": stage_output,
            "status": "waiting_for_user",
        }

    def _get_extractor_agent(self) -> Any:
        if self.extractor_agent is None:
            self.extractor_agent = AgentFactory.build_agent(
                prompt=EXTRACTOR_PROMPT,
                tools=[],
                temperature=self.temperature,
                model_name=self.model_name,
                response_format=RequirementsExtractorOutput,
            )
        return self.extractor_agent

    def _get_critic_agent(self) -> Any:
        if self.critic_agent is None:
            self.critic_agent = AgentFactory.build_agent(
                prompt=CRITIC_PROMPT,
                tools=[],
                temperature=self.temperature,
                model_name=self.model_name,
                response_format=RequirementsCriticOutput,
            )
        return self.critic_agent

    @staticmethod
    def _to_result(state: RequirementsState) -> RequirementsRunResult:
        extractor_output = state.get("extractor_output")
        critic_output = state.get("critic_output")
        return RequirementsRunResult(
            status=state.get("status", "waiting_for_user"),
            requirements=extractor_output.requirements if extractor_output else None,
            question_answer_context=state.get("question_answer_context", []),
            critic_output=critic_output,
            stage_output=state.get("stage_output"),
            questions=critic_output.questions if critic_output else [],
        )


def build_requirements_graph(**kwargs: Any) -> RequirementsStageGraph:
    return RequirementsStageGraph(**kwargs)


def _run_structured_agent(agent: Any, agent_input: BaseModel) -> Any:
    payload = agent_input.model_dump(mode="json")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=json.dumps(payload, indent=2),
                )
            ]
        }
    )
    return result


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

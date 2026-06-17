from __future__ import annotations

import json
from typing import Any, Callable, Literal, TypedDict, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context
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
from src.settings import configure_langsmith_environment, settings


RequirementsStatus = Literal["complete", "waiting_for_user", "failed"]
AgentRunner = Callable[[Any, BaseModel], Any]
ModelT = TypeVar("ModelT", bound=BaseModel)

from typing import TypedDict, List

class InternalMetrics(TypedDict):
    number_of_iterations: int
    number_of_questions_asked_per_iteration: List[int] # Or just list

# import  txt file
def load_txt_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

CRITIC_PROMPT = load_txt_file("src/multi_agent_system/prompts/RequirementsCritic.txt").strip()
EXTRACTOR_PROMPT = load_txt_file("src/multi_agent_system/prompts/RequirementsExtractor.txt").strip()


class RequirementsState(TypedDict, total=False):
    project_prompt: str
    previous_requirements: RequirementsSpec | None
    question_answer_context: list[QuestionAnswer]
    extractor_output: RequirementsExtractorOutput
    critic_output: RequirementsCriticOutput
    stage_output: RequirementsStageOutput
    status: RequirementsStatus
    failure_reason: str
    force_continue_reason: str
    def __str__(self):        return (
            f"Project Prompt: {self.project_prompt}\n\n" +
            f"Previous Requirements: \n{self.previous_requirements}\n\n" +
            f"Question-Answer Context: \n{self.question_answer_context}\n\n" +
            f"Extractor Output: \n{self.extractor_output}\n\n" +
            f"Critic Output: \n{self.critic_output}\n\n" +
            f"Stage Output: \n{self.stage_output}\n\n" +
            f"Status: {self.status}"
        )


class RequirementsRunResult(BaseModel):
    status: RequirementsStatus
    requirements: RequirementsSpec | None = None
    question_answer_context: list[QuestionAnswer] = Field(default_factory=list)
    critic_output: RequirementsCriticOutput | None = None
    stage_output: RequirementsStageOutput | None = None
    questions: list[str] = Field(default_factory=list)
    failure_reason: str = ""

    def __str__(self):
        return (
            f"Status: {self.status}\n\n" +
            f"Requirements: \n{self.requirements}\n\n" +
            f"Question-Answer Context: {self.question_answer_context}\n\n" +
            f"Critic Output: {self.critic_output}\n\n" +
            f"Stage Output: {self.stage_output}\n\n" +
            f"Questions from Critic: {self.questions}"
        )


class RequirementsStageGraph:
    """Small LangGraph for requirements extraction and Critic A review."""

    def __init__(
        self,
        extractor_agent: Any | None = None,
        critic_agent: Any | None = None,
        extractor_runner: AgentRunner | None = None,
        critic_runner: AgentRunner | None = None,
        model_name: str = "gpt-5.4-mini",
        temperature: float = 0.2,
        max_iterations: int = 2,
    ) -> None:
        print("Initializing RequirementsStageGraph...")
        self.extractor_agent = extractor_agent
        self.critic_agent = critic_agent
        self.extractor_runner = extractor_runner or _run_structured_agent
        self.critic_runner = critic_runner or _run_structured_agent
        self.model_name = model_name
        self.temperature = temperature
        self.graph = self._build_requirement_graph()
        self.internal_metrics: InternalMetrics = {
            "number_of_iterations": 0,
            "number_of_questions_asked_per_iteration": [],}
        self.max_iterations = max_iterations

    def run(
        self,
        project_prompt: str,
        previous_requirements: RequirementsSpec | None = None,
        question_answer_context: list[QuestionAnswer] | None = None,
    ) -> RequirementsRunResult:
        configure_langsmith_environment(settings)
        print("Running RequirementsStageGraph...")
        initial_state: RequirementsState = {
            "project_prompt": project_prompt,
            "previous_requirements": previous_requirements,
            "question_answer_context": question_answer_context or [],
        }
        trace_tags = ["requirements_graph", f"model:{self.model_name}"]
        trace_metadata = {
            "graph": "requirements",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "has_previous_requirements": previous_requirements is not None,
            "question_count": len(question_answer_context or []),
        }
        with tracing_context(
            enabled=settings.LANGSMITH_TRACING,
            project_name=settings.LANGSMITH_PROJECT,
            tags=trace_tags,
            metadata=trace_metadata,
        ):
            final_state = self.graph.invoke(initial_state)
        return self._to_result(final_state)

    def _build_requirement_graph(self):
        print("Building requirements graph...")
        graph_builder = StateGraph(RequirementsState)
        graph_builder.add_node("extract_requirements", self.extract_requirements)
        graph_builder.add_node("critic_review", self.critic_review)
        graph_builder.add_node("complete", self.complete)
        graph_builder.add_node("waiting_for_user", self.waiting_for_user)
        graph_builder.add_node("failed", self.failed)

        graph_builder.add_edge(START, "extract_requirements")
        graph_builder.add_edge("extract_requirements", "critic_review")
        graph_builder.add_conditional_edges(
            "critic_review",
            self.route_after_critic,
            {
                "complete": "complete",
                "waiting_for_user": "waiting_for_user",
                "failed": "failed",
            },
        )
        graph_builder.add_edge("complete", END)
        graph_builder.add_edge("waiting_for_user", END)
        graph_builder.add_edge("failed", END)
        graph = graph_builder.compile()
    
        return graph

    def extract_requirements(self, state: RequirementsState) -> RequirementsState:
        print("Node: extract_requirements")
        self.internal_metrics["number_of_iterations"] += 1
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
        print("*"*20)
        print("Extracted Requirements:", extractor_output.requirements)
        return {
            **state,
            "extractor_output": extractor_output,
            "previous_requirements": extractor_output.requirements,
        }

    def critic_review(self, state: RequirementsState) -> RequirementsState:
        print("Node: critic_review")
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
        self.internal_metrics["number_of_questions_asked_per_iteration"].append(len(critic_output.questions))
        if self.internal_metrics["number_of_iterations"] > self.max_iterations:
            print("=========================\nWarning: Number of iterations exceeded. Max iterations:", self.max_iterations)
            print("=========================")
            print("Continuing to the next stage with the current requirements.")
            return {
                **state,
                "critic_output": critic_output,
                "force_continue_reason": f"Maximum requirements iterations exceeded: {self.max_iterations}",
            }
        return {
            **state,
            "critic_output": critic_output,
        }

    @staticmethod
    def route_after_critic(state: RequirementsState) -> str:
        print("Node: route_after_critic")
        critic_output = state["critic_output"]
        print("Critic approved?", critic_output.approved)


        if state.get("force_continue_reason"):
            return "complete"
        if state.get("failure_reason"):
            return "failed"
        if critic_output.questions:
            return "waiting_for_user"
        if critic_output.approved:
            return "complete"
        return "failed"

    @staticmethod
    def complete(state: RequirementsState) -> RequirementsState:
        print("Node: complete")
        extractor_output = state["extractor_output"]
        critic_output = state["critic_output"]
        stage_output = RequirementsStageOutput(
            requirements=extractor_output.requirements,
            question_answer_context=state.get("question_answer_context", []),
            critic_verdict=_build_requirements_verdict(state, critic_output),
            approved=True,
        )
        res = {
            **state,
            "stage_output": stage_output,
            "status": "complete",
        }
        # print("Final state at complete node:", state)
        print("Keys in state at complete node:", list(state.keys()))
        return res


    @staticmethod
    def waiting_for_user(state: RequirementsState) -> RequirementsState:
        print("Node: waiting_for_user")
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

    @staticmethod
    def failed(state: RequirementsState) -> RequirementsState:
        print("Node: failed")
        extractor_output = state["extractor_output"]
        critic_output = state["critic_output"]
        stage_output = RequirementsStageOutput(
            requirements=extractor_output.requirements,
            question_answer_context=state.get("question_answer_context", []),
            critic_verdict=critic_output.verdict,
            approved=False,
        )
        failure_reason = state.get("failure_reason") or (
            "Requirements critic did not approve and did not return user questions."
        )
        return {
            **state,
            "stage_output": stage_output,
            "status": "failed",
            "failure_reason": failure_reason,
        }

    def _get_extractor_agent(self) -> Any:
        if self.extractor_agent is None:
            print("Initializing extractor agent...")

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
            print("Initializing critic agent...")

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
            failure_reason=state.get("failure_reason", ""),
        )
    @staticmethod
    def print_final_output(state: RequirementsState) -> None:
        print("\nFinal Output:")
        print("Status:", state.get("status"))
        if "stage_output" in state:
            stage_output = state["stage_output"]
            print("Approved:", stage_output.approved)
            print("Critic Verdict:", stage_output.critic_verdict)
            print("Extracted Requirements:", stage_output.requirements)
            print("Question-Answer Context:", stage_output.question_answer_context)

def build_requirements_graph(**kwargs: Any) -> RequirementsStageGraph:
    return RequirementsStageGraph(**kwargs)


def _build_requirements_verdict(
    state: RequirementsState,
    critic_output: RequirementsCriticOutput,
) -> str:
    force_continue_reason = state.get("force_continue_reason")
    if force_continue_reason:
        return f"{critic_output.verdict}\n\nForce continued: {force_continue_reason}"
    return critic_output.verdict


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

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from src.agents.human_approval_runner import run_with_human_approval


Status = Literal["running", "success", "failed_needs_human"]
AgentRunner = Callable[[Any, str, dict[str, Any]], dict[str, Any]]


class ExecutorWriterState(TypedDict, total=False):
    user_task: str
    testing_framework: str
    attempt: int
    max_attempts: int
    last_executor_output: str | None
    writer_state: dict[str, Any]
    executor_state: dict[str, Any]
    status: Status


class ExecutorWriterGraph:
    """Coordinate a writer agent and executor agent with a LangGraph retry loop."""

    def __init__(
        self,
        writer_agent: Any | None = None,
        executor_agent: Any | None = None,
        writer_runner: AgentRunner = run_with_human_approval,
        executor_runner: AgentRunner = run_with_human_approval,
    ) -> None:
        self.writer_agent = writer_agent
        self.executor_agent = executor_agent
        self.writer_runner = writer_runner
        self.executor_runner = executor_runner
        self.writer_config = {"configurable": {"thread_id": f"writer-{uuid.uuid4()}"}}
        self.executor_config = {"configurable": {"thread_id": f"executor-{uuid.uuid4()}"}}
        self.graph = self._build_graph()

    def run(
        self,
        user_task: str,
        testing_framework: str,
        max_attempts: int = 5,
    ) -> dict[str, Any]:
        if not user_task.strip():
            raise ValueError("user_task is required")
        if not testing_framework.strip():
            raise ValueError("testing_framework is required")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        initial_state: ExecutorWriterState = {
            "user_task": user_task,
            "testing_framework": testing_framework,
            "attempt": 0,
            "max_attempts": max_attempts,
            "last_executor_output": None,
            "writer_state": {},
            "executor_state": {},
            "status": "running",
        }
        final_state = self.graph.invoke(initial_state)

        return {
            "status": final_state["status"],
            "attempts": final_state["attempt"],
            "writer_state": final_state.get("writer_state", {}),
            "executor_state": final_state.get("executor_state", {}),
            "final_executor_output": final_state.get("last_executor_output"),
        }

    def _build_graph(self):
        graph_builder = StateGraph(ExecutorWriterState)
        graph_builder.add_node("writer", self.writer_node)
        graph_builder.add_node("executor", self.executor_node)
        graph_builder.add_node("human_review", self.human_review_node)

        graph_builder.add_edge(START, "writer")
        graph_builder.add_edge("writer", "executor")
        graph_builder.add_conditional_edges(
            "executor",
            self._route_after_executor,
            {
                "success": END,
                "retry": "writer",
                "human_review": "human_review",
            },
        )
        graph_builder.add_edge("human_review", END)

        return graph_builder.compile()

    def writer_node(self, state: ExecutorWriterState) -> ExecutorWriterState:
        writer_agent = self._get_writer_agent()
        prompt = self._build_writer_prompt(state)
        writer_state = self.writer_runner(writer_agent, prompt, self.writer_config)

        return {
            **state,
            "writer_state": writer_state,
            "status": "running",
        }

    def executor_node(self, state: ExecutorWriterState) -> ExecutorWriterState:
        executor_agent = self._get_executor_agent()
        prompt = self._build_executor_prompt(state)
        executor_state = self.executor_runner(executor_agent, prompt, self.executor_config)
        executor_output = self._extract_executor_output(executor_state)
        attempt = state.get("attempt", 0) + 1
        status: Status = "success" if self._executor_succeeded(executor_output) else "running"

        return {
            **state,
            "attempt": attempt,
            "executor_state": executor_state,
            "last_executor_output": executor_output,
            "status": status,
        }

    def human_review_node(self, state: ExecutorWriterState) -> ExecutorWriterState:
        return {
            **state,
            "status": "failed_needs_human",
        }

    def _route_after_executor(self, state: ExecutorWriterState) -> str:
        if state.get("status") == "success":
            return "success"

        if state.get("attempt", 0) >= state.get("max_attempts", 5):
            return "human_review"

        return "retry"

    def _get_writer_agent(self) -> Any:
        if self.writer_agent is None:
            from src.agents.writer.agent import build_writer_agent

            self.writer_agent = build_writer_agent()

        return self.writer_agent

    def _get_executor_agent(self) -> Any:
        if self.executor_agent is None:
            from src.agents.code_executor.agent import build_code_executor_agent

            self.executor_agent = build_code_executor_agent()

        return self.executor_agent

    @staticmethod
    def _build_writer_prompt(state: ExecutorWriterState) -> str:
        prompt = (
            "You are the writer agent in an executor-writer subsystem.\n"
            "Implement the user's task and write or update tests using the provided testing framework.\n"
            "Do not execute code; the executor agent will choose and run verification commands.\n\n"
            f"User task:\n{state['user_task']}\n\n"
            f"Testing framework:\n{state['testing_framework']}\n"
        )

        if state.get("last_executor_output"):
            prompt += (
                "\nThe executor found errors in the previous attempt. "
                "Use this output to repair the code or tests:\n"
                f"{state['last_executor_output']}\n"
            )

        return prompt

    @staticmethod
    def _build_executor_prompt(state: ExecutorWriterState) -> str:
        return (
            "You are the executor agent in an executor-writer subsystem.\n"
            "Inspect the repository, choose the appropriate command for the provided testing framework, "
            "and run verification. Do not edit files.\n\n"
            f"User task:\n{state['user_task']}\n\n"
            f"Testing framework:\n{state['testing_framework']}\n\n"
            "Return tool output that clearly indicates whether execution succeeded or failed."
        )

    @staticmethod
    def _extract_executor_output(executor_state: dict[str, Any]) -> str | None:
        messages = executor_state.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                content = _message_content(message)
                if content:
                    return content

        for key in ("output", "result", "final_executor_output"):
            value = executor_state.get(key)
            if value is not None:
                return str(value)

        return None

    @staticmethod
    def _executor_succeeded(executor_output: str | None) -> bool:
        if not executor_output:
            return False

        try:
            parsed = json.loads(executor_output)
        except json.JSONDecodeError:
            return "success" in executor_output.lower() and "error" not in executor_output.lower()

        if not isinstance(parsed, dict):
            return False

        if parsed.get("status") == "success":
            return True

        exit_code = parsed.get("exit_code")
        return isinstance(exit_code, int) and exit_code == 0


def _message_content(message: Any) -> str | None:
    if isinstance(message, ToolMessage):
        return str(message.content)

    if isinstance(message, BaseMessage):
        return str(message.content)

    if isinstance(message, dict) and message.get("content") is not None:
        return str(message["content"])

    return None


if __name__ == "__main__":
    graph = ExecutorWriterGraph()
    print("Executor-writer graph created successfully.")

    while True:
        user_task = input("User task (or 'exit' to quit): ").strip()
        if user_task.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        testing_framework = input("Testing framework: ").strip()
        if not testing_framework:
            print("Testing framework is required.")
            continue

        result = graph.run(user_task=user_task, testing_framework=testing_framework)
        print("\nExecutor-writer result:")
        print(json.dumps(result, indent=2, default=str))

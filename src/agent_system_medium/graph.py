from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from src.agent_system_medium.schemas import (
    AgentProgressUpdate,
    ApprovalDecision,
    ExecutorCommandResult,
    ExecutorOutput,
    FileTask,
    FileTaskProgress,
    MediumGraphResult,
    SupervisorOutput,
    TraceEvent,
    WriterInput,
    WriterOutput,
)
from src.agents.human_approval_runner import run_with_human_approval


AgentRunner = Callable[[Any, str, dict[str, Any]], dict[str, Any]]
ApprovalCallback = Callable[[str, dict[str, Any], dict[str, Any]], ApprovalDecision]
ProgressCallback = Callable[[str, "MediumAgentState"], None]
Status = Literal["running", "success", "failed_needs_human"]
GraphTopology = Literal["supervisor_centered", "writer_executor"]


class MediumAgentState(TypedDict, total=False):
    user_prompt: str
    max_repair_attempts: int
    max_revision_attempts: int
    repair_attempts: int
    revision_attempts: dict[str, int]
    supervisor_output: SupervisorOutput
    writer_outputs: list[WriterOutput]
    executor_output: ExecutorOutput | None
    progress_update: AgentProgressUpdate
    task_store: list[FileTaskProgress]
    current_file_task_index: int
    last_executor_output: str | None
    current_agent: str | None
    trace: list[TraceEvent]
    status: Status
    route_after_approval: str


def auto_approve(agent_name: str, step_input: dict[str, Any], step_output: dict[str, Any]) -> ApprovalDecision:
    return ApprovalDecision(approved=True, message="Auto-approved.")


def interactive_approve(agent_name: str, step_input: dict[str, Any], step_output: dict[str, Any]) -> ApprovalDecision:
    print(f"\nApproval required for {agent_name}")
    print("Input:")
    print(json.dumps(step_input, indent=2, default=str))
    print("Output:")
    print(json.dumps(step_output, indent=2, default=str))

    while True:
        choice = input("Approve this agent step? [y/n]: ").strip().lower()
        if choice in {"y", "yes"}:
            return ApprovalDecision(approved=True)
        if choice in {"n", "no"}:
            message = input("Revision note or rejection reason: ").strip()
            return ApprovalDecision(approved=False, message=message)
        print("Please answer y or n.")


def print_agent_progress(agent_name: str, state: MediumAgentState) -> None:
    print(f"[agent_system_medium] running agent: {agent_name}")


def quiet_agent_progress(agent_name: str, state: MediumAgentState) -> None:
    return None


class AgentSystemMediumGraph:
    """Supervisor-led medium SDLC system."""

    def __init__(
        self,
        supervisor_agent: Any | None = None,
        writer_agent: Any | None = None,
        executor_agent: Any | None = None,
        supervisor_runner: AgentRunner = run_with_human_approval,
        writer_runner: AgentRunner = run_with_human_approval,
        executor_runner: AgentRunner = run_with_human_approval,
        approval_callback: ApprovalCallback = interactive_approve,
        progress_callback: ProgressCallback | None = print_agent_progress,
        graph_topology: GraphTopology = "supervisor_centered",
    ) -> None:
        self.supervisor_agent = supervisor_agent
        self.writer_agent = writer_agent
        self.executor_agent = executor_agent
        self.supervisor_runner = supervisor_runner
        self.writer_runner = writer_runner
        self.executor_runner = executor_runner
        self.approval_callback = approval_callback
        self.progress_callback = progress_callback
        self.graph_topology = graph_topology
        self.supervisor_config = {"configurable": {"thread_id": f"supervisor-{uuid.uuid4()}"}}
        self.writer_config = {"configurable": {"thread_id": f"writer-{uuid.uuid4()}"}}
        self.executor_config = {"configurable": {"thread_id": f"executor-{uuid.uuid4()}"}}
        self.graph = self._build_graph(graph_topology)

    @classmethod
    def supervisor_centered(cls, **kwargs: Any) -> "AgentSystemMediumGraph":
        kwargs["graph_topology"] = "supervisor_centered"
        return cls(**kwargs)

    @classmethod
    def writer_executor(cls, **kwargs: Any) -> "AgentSystemMediumGraph":
        kwargs["graph_topology"] = "writer_executor"
        return cls(**kwargs)

    def run(
        self,
        user_prompt: str,
        max_repair_attempts: int = 5,
        max_revision_attempts: int = 3,
    ) -> dict[str, Any]:
        if not user_prompt.strip():
            raise ValueError("user_prompt is required")
        if max_repair_attempts < 1:
            raise ValueError("max_repair_attempts must be at least 1")
        if max_revision_attempts < 1:
            raise ValueError("max_revision_attempts must be at least 1")

        initial_state: MediumAgentState = {
            "user_prompt": user_prompt,
            "max_repair_attempts": max_repair_attempts,
            "max_revision_attempts": max_revision_attempts,
            "repair_attempts": 0,
            "revision_attempts": {},
            "writer_outputs": [],
            "executor_output": None,
            "progress_update": AgentProgressUpdate(
                stage="start",
                status="not complete",
                files_written_or_commands_executed=[],
            ),
            "task_store": [],
            "current_file_task_index": 0,
            "last_executor_output": None,
            "current_agent": None,
            "trace": [],
            "status": "running",
        }
        final_state = self.graph.invoke(initial_state)
        supervisor_output = final_state.get("supervisor_output") or SupervisorOutput()
        result = MediumGraphResult(
            status=final_state.get("status", "failed_needs_human"),
            attempts=final_state.get("repair_attempts", 0),
            current_agent=final_state.get("current_agent"),
            progress_update=final_state.get("progress_update"),
            task_store=final_state.get("task_store", []),
            supervisor_output=supervisor_output,
            writer_outputs=final_state.get("writer_outputs", []),
            executor_output=final_state.get("executor_output"),
            trace=final_state.get("trace", []),
        )
        return result.model_dump()

    def _build_graph(self, graph_topology: GraphTopology = "supervisor_centered"):
        if graph_topology == "writer_executor":
            return self._build_writer_executor_graph()
        return self._build_supervisor_centered_graph()

    def _build_supervisor_centered_graph(self):
        graph_builder = StateGraph(MediumAgentState)
        graph_builder.add_node("supervisor", self.supervisor_node)
        graph_builder.add_node("writer", self.writer_node)
        graph_builder.add_node("executor", self.executor_node)
        graph_builder.add_node("human_review", self.human_review_node)

        graph_builder.add_edge(START, "supervisor")
        graph_builder.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {
                "success": END,
                "writer": "writer",
                "executor": "executor",
                "supervisor": "supervisor",
                "human_review": "human_review",
            },
        )
        graph_builder.add_conditional_edges(
            "writer",
            self._route_after_writer_to_supervisor,
            {
                "writer": "writer",
                "supervisor": "supervisor",
                "human_review": "human_review",
            },
        )
        graph_builder.add_conditional_edges(
            "executor",
            self._route_after_executor_to_supervisor,
            {
                "executor": "executor",
                "supervisor": "supervisor",
                "human_review": "human_review",
            },
        )
        graph_builder.add_edge("human_review", END)
        return graph_builder.compile()

    def _build_writer_executor_graph(self):
        graph_builder = StateGraph(MediumAgentState)
        graph_builder.add_node("supervisor", self.supervisor_node)
        graph_builder.add_node("writer", self.writer_node)
        graph_builder.add_node("executor", self.executor_node)
        graph_builder.add_node("human_review", self.human_review_node)

        graph_builder.add_edge(START, "supervisor")
        graph_builder.add_conditional_edges(
            "supervisor",
            self._route_after_initial_supervisor,
            {
                "writer": "writer",
                "supervisor": "supervisor",
                "human_review": "human_review",
            },
        )
        graph_builder.add_conditional_edges(
            "writer",
            self._route_after_writer,
            {
                "writer": "writer",
                "executor": "executor",
                "human_review": "human_review",
            },
        )
        graph_builder.add_conditional_edges(
            "executor",
            self._route_after_executor,
            {
                "success": END,
                "writer": "writer",
                "executor": "executor",
                "human_review": "human_review",
            },
        )
        graph_builder.add_edge("human_review", END)
        return graph_builder.compile()

    def supervisor_node(self, state: MediumAgentState) -> MediumAgentState:
        self._announce_agent("supervisor", state)
        agent = self._get_supervisor_agent()
        step_input = {
            "stage": state.get("progress_update", AgentProgressUpdate(stage="start", status="not complete")).stage,
            "status": state.get("progress_update", AgentProgressUpdate(stage="start", status="not complete")).status,
            "files_written_or_commands_executed": state.get(
                "progress_update",
                AgentProgressUpdate(stage="start", status="not complete"),
            ).files_written_or_commands_executed,
            "user_prompt": state["user_prompt"],
            "task_store": [task.model_dump() for task in state.get("task_store", [])],
            "previous_feedback": self._latest_rejection_message(state, "supervisor"),
            "completed_writer_outputs": [output.model_dump() for output in state.get("writer_outputs", [])],
            "latest_executor_output": state.get("executor_output").model_dump()
            if state.get("executor_output")
            else None,
            "repair_attempts": state.get("repair_attempts", 0),
        }
        try:
            raw_state = self.supervisor_runner(agent, self._build_supervisor_prompt(step_input), self.supervisor_config)
        except Exception as error:
            message = _runner_error_message("supervisor", error)
            output = SupervisorOutput(project_summary=message)
            approval = ApprovalDecision(approved=False, message=message)
            trace = self._append_trace(state, "supervisor", "plan_error", step_input, output.model_dump(), approval)
            next_state: MediumAgentState = {
                **state,
                "current_agent": "supervisor",
                "supervisor_output": state.get("supervisor_output") or output,
                "trace": trace,
                "route_after_approval": "supervisor",
                "last_executor_output": message,
            }
            return self._increment_revision(next_state, "supervisor")
        output = self._extract_supervisor_output(raw_state)
        approval = self.approval_callback("supervisor", step_input, output.model_dump())
        trace = self._append_trace(state, "supervisor", "plan", step_input, output.model_dump(), approval)
        previous_supervisor_output = state.get("supervisor_output")
        if previous_supervisor_output and state.get("task_store"):
            output.file_tasks = previous_supervisor_output.file_tasks
        elif previous_supervisor_output and not output.file_tasks:
            output.file_tasks = previous_supervisor_output.file_tasks
        task_store = state.get("task_store", [])
        if approval.approved and not task_store:
            task_store = self._build_task_store(self._file_tasks_or_default(output))

        next_state: MediumAgentState = {
            **state,
            "current_agent": "supervisor",
            "supervisor_output": output,
            "task_store": task_store,
            "trace": trace,
            "route_after_approval": "writer" if approval.approved else "supervisor",
        }
        if not approval.approved:
            next_state = self._increment_revision(next_state, "supervisor")
        return next_state

    def writer_node(self, state: MediumAgentState) -> MediumAgentState:
        self._announce_agent("writer", state)
        supervisor_output = state.get("supervisor_output") or SupervisorOutput()
        file_tasks = self._file_tasks_or_default(supervisor_output)
        task_store = state.get("task_store", []) or self._build_task_store(file_tasks)
        task_index = self._next_pending_task_index(task_store)
        if task_index is None:
            task_index = min(state.get("current_file_task_index", 0), len(file_tasks) - 1)
        file_task = file_tasks[task_index]
        progress_update = state.get("progress_update", AgentProgressUpdate(stage="start", status="not complete"))
        writer_input = WriterInput(
            user_prompt=state["user_prompt"],
            progress_update=progress_update,
            task_store=task_store,
            supervisor_output=supervisor_output,
            file_task=file_task,
            previous_executor_output=state.get("last_executor_output"),
        )

        try:
            raw_state = self.writer_runner(
                self._get_writer_agent(),
                self._build_writer_prompt(writer_input),
                self.writer_config,
            )
            writer_output = WriterOutput(
                task_id=file_task.task_id,
                relative_path=file_task.relative_path,
                status="done_untested",
                summary=_extract_agent_text(raw_state) or "Writer step completed.",
                raw_agent_state=raw_state,
            )
            approval = self.approval_callback(
                "writer",
                writer_input.model_dump(),
                writer_output.model_dump(),
            )
        except Exception as error:
            message = _runner_error_message("writer", error)
            writer_output = WriterOutput(
                task_id=file_task.task_id,
                relative_path=file_task.relative_path,
                status="failed",
                summary=message,
                raw_agent_state={"error": message, "exception_type": type(error).__name__},
            )
            approval = ApprovalDecision(approved=False, message=message)
        trace = self._append_trace(
            state,
            "writer",
            f"file_task:{file_task.task_id}",
            writer_input.model_dump(),
            writer_output.model_dump(),
            approval,
        )

        next_state: MediumAgentState = {
            **state,
            "current_agent": "writer",
            "trace": trace,
            "route_after_approval": "writer" if not approval.approved else "next",
        }
        if approval.approved:
            writer_outputs = [*state.get("writer_outputs", []), writer_output]
            next_state["writer_outputs"] = writer_outputs
            next_state["task_store"] = self._update_task_status(task_store, file_task.task_id, "written_untested")
            next_state["progress_update"] = AgentProgressUpdate(
                stage="coding",
                status="not complete",
                files_written_or_commands_executed=[file_task.relative_path],
            )
            next_pending_index = self._next_pending_task_index(next_state["task_store"])
            next_state["current_file_task_index"] = next_pending_index if next_pending_index is not None else task_index + 1
            next_state["executor_output"] = None
        else:
            next_state["writer_outputs"] = [*state.get("writer_outputs", []), writer_output]
            next_state["task_store"] = self._update_task_status(task_store, file_task.task_id, "failed")
            next_state["last_executor_output"] = writer_output.summary
            next_state["progress_update"] = AgentProgressUpdate(
                stage="coding",
                status="not complete",
                files_written_or_commands_executed=[writer_output.summary],
            )
            next_state["current_file_task_index"] = task_index
            next_state = self._increment_revision(next_state, "writer")
        return next_state

    def executor_node(self, state: MediumAgentState) -> MediumAgentState:
        self._announce_agent("executor", state)
        supervisor_output = state.get("supervisor_output") or SupervisorOutput()
        progress_update = state.get("progress_update", AgentProgressUpdate(stage="start", status="not complete"))
        step_input = {
            "stage": "execution",
            "status": "not complete",
            "files_written_or_commands_executed": progress_update.files_written_or_commands_executed,
            "user_prompt": state["user_prompt"],
            "task_store": [task.model_dump() for task in state.get("task_store", [])],
            "supervisor_output": supervisor_output.model_dump(),
            "last_executor_output": state.get("last_executor_output"),
        }
        try:
            raw_state = self.executor_runner(
                self._get_executor_agent(),
                self._build_executor_prompt(step_input),
                self.executor_config,
            )
            executor_output = self._extract_executor_output(raw_state)
            approval = self.approval_callback("executor", step_input, executor_output.model_dump())
        except Exception as error:
            message = _runner_error_message("executor", error)
            executor_output = ExecutorOutput(
                status="error",
                command_results=[
                    ExecutorCommandResult(
                        command_phase="test",
                        command=[],
                        status="error",
                        exit_code=None,
                        stderr=message,
                        message=message,
                    )
                ],
                summary=message,
                raw_agent_state={"error": message, "exception_type": type(error).__name__},
            )
            approval = ApprovalDecision(approved=True, message="Runner error relayed to repair loop.")
        trace = self._append_trace(
            state,
            "executor",
            "setup_and_test",
            step_input,
            executor_output.model_dump(),
            approval,
        )

        next_state: MediumAgentState = {
            **state,
            "current_agent": "executor",
            "executor_output": executor_output,
            "last_executor_output": executor_output.summary,
            "trace": trace,
            "route_after_approval": "next" if approval.approved else "executor",
        }
        if not approval.approved:
            next_state = self._increment_revision(next_state, "executor")
        elif executor_output.status == "error":
            commands = self._executor_commands_or_summary(executor_output)
            next_state["repair_attempts"] = state.get("repair_attempts", 0) + 1
            next_state["task_store"] = self._reset_written_tasks_for_repair(state.get("task_store", []))
            next_state["progress_update"] = AgentProgressUpdate(
                stage="execution",
                status="not complete",
                files_written_or_commands_executed=commands,
            )
            next_pending_index = self._next_pending_task_index(next_state["task_store"])
            next_state["current_file_task_index"] = next_pending_index if next_pending_index is not None else 0
        else:
            commands = self._executor_commands_or_summary(executor_output)
            next_state["task_store"] = self._mark_written_tasks_tested(state.get("task_store", []))
            next_state["progress_update"] = AgentProgressUpdate(
                stage="execution",
                status="complete",
                files_written_or_commands_executed=commands,
            )
            next_state["status"] = "success"
        return next_state

    def human_review_node(self, state: MediumAgentState) -> MediumAgentState:
        self._announce_agent("human_review", state)
        return {
            **state,
            "current_agent": "human_review",
            "status": "failed_needs_human",
        }

    def _route_after_supervisor(self, state: MediumAgentState) -> str:
        if self._revision_limit_reached(state, "supervisor"):
            return "human_review"
        if state.get("route_after_approval") != "writer":
            return "supervisor"

        if state.get("status") == "success":
            return "success"

        progress_update = state.get("progress_update")
        if (
            progress_update
            and progress_update.stage == "execution"
            and progress_update.status == "not complete"
            and state.get("repair_attempts", 0) >= state.get("max_repair_attempts", 5)
        ):
            return "human_review"

        task_store = state.get("task_store", [])
        if task_store and all(task.status == "tested" for task in task_store):
            return "success"
        if task_store and any(task.status in {"pending", "failed"} for task in task_store):
            return "writer"
        if task_store and any(task.status == "written_untested" for task in task_store):
            return "executor"

        executor_output = state.get("executor_output")
        if executor_output and executor_output.status == "success":
            return "success"
        if executor_output and executor_output.status == "error":
            if state.get("repair_attempts", 0) >= state.get("max_repair_attempts", 5):
                return "human_review"
            return "writer"

        supervisor_output = state.get("supervisor_output") or SupervisorOutput()
        file_task_count = len(self._file_tasks_or_default(supervisor_output))
        if state.get("current_file_task_index", 0) >= file_task_count:
            return "executor"
        return "writer"

    def _route_after_initial_supervisor(self, state: MediumAgentState) -> str:
        if state.get("route_after_approval") == "writer":
            return "writer"
        if self._revision_limit_reached(state, "supervisor"):
            return "human_review"
        return "supervisor"

    def _route_after_writer(self, state: MediumAgentState) -> str:
        if state.get("route_after_approval") == "writer":
            if self._revision_limit_reached(state, "writer"):
                return "human_review"
            return "writer"

        supervisor_output = state.get("supervisor_output") or SupervisorOutput()
        if state.get("current_file_task_index", 0) >= len(self._file_tasks_or_default(supervisor_output)):
            return "executor"
        return "writer"

    def _route_after_writer_to_supervisor(self, state: MediumAgentState) -> str:
        if state.get("route_after_approval") == "writer":
            if self._revision_limit_reached(state, "writer"):
                return "human_review"
            return "writer"
        return "supervisor"

    def _route_after_executor(self, state: MediumAgentState) -> str:
        if state.get("status") == "success":
            return "success"
        if state.get("route_after_approval") == "executor":
            if self._revision_limit_reached(state, "executor"):
                return "human_review"
            return "executor"
        if state.get("repair_attempts", 0) >= state.get("max_repair_attempts", 5):
            return "human_review"
        return "writer"

    def _route_after_executor_to_supervisor(self, state: MediumAgentState) -> str:
        if state.get("route_after_approval") == "executor":
            if self._revision_limit_reached(state, "executor"):
                return "human_review"
            return "executor"
        return "supervisor"

    def _get_supervisor_agent(self) -> Any:
        if self.supervisor_agent is None:
            from src.agents.supervisor_medium.agent import build_supervisor_agent

            self.supervisor_agent = build_supervisor_agent()
        return self.supervisor_agent

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
    def _build_supervisor_prompt(step_input: dict[str, Any]) -> str:
        return (
            "You are the Supervisor Agent. Create or update the structured SDLC plan for this project prompt. "
            "The input always contains stage, status, and files_written_or_commands_executed. "
            "When stage is start, produce the initial plan and file tasks. "
            "When stage is coding, inspect the written-but-untested file progress and decide which remaining file should be written next. "
            "When stage is execution, inspect the command/test result and decide whether the written files are complete or need repair. "
            "Use task_store as the persistent memory of which files are pending, written_untested, tested, or failed; do not rely on memory alone. "
            "Include requirements, design, repository structure, file tasks, dependency plan, and test plan.\n\n"
            "Format dependency_plan.setup_commands and test_plan.test_commands as arrays of argv arrays, "
            'for example [["python", "-m", "pytest"]]. Do not use command/description objects for commands.\n\n'
            f"Input:\n{json.dumps(step_input, indent=2, default=str)}"
        )

    @staticmethod
    def _build_writer_prompt(writer_input: WriterInput) -> str:
        return (
            "You are the Writer Agent. Implement exactly this file task using the approved supervisor plan. "
            "Create or edit files only. Do not execute commands. "
            "When finished, report that the file is done but untested; testing belongs to the Executor Agent.\n\n"
            f"Structured input:\n{writer_input.model_dump_json(indent=2)}"
        )

    @staticmethod
    def _build_executor_prompt(step_input: dict[str, Any]) -> str:
        return (
            "You are the Executor Agent. Use the approved supervisor plan to understand dependencies, setup commands, "
            "test framework, and execution commands. Repository command tools already run inside the configured project repository. "
            "Start or reuse a persistent container, run setup commands first, "
            "then run test/execution commands. Do not edit files. "
            "Report which commands were executed and whether execution is complete or not complete.\n\n"
            f"Structured input:\n{json.dumps(step_input, indent=2, default=str)}"
        )

    @staticmethod
    def _extract_supervisor_output(raw_state: dict[str, Any]) -> SupervisorOutput:
        structured = raw_state.get("structured_response")
        if isinstance(structured, SupervisorOutput):
            return structured
        if isinstance(structured, dict):
            return SupervisorOutput.model_validate(structured)

        text = _extract_agent_text(raw_state)
        if text:
            try:
                return SupervisorOutput.model_validate_json(text)
            except ValueError:
                pass

        return SupervisorOutput(
            project_summary=text or "Supervisor did not return a structured plan.",
            file_tasks=[
                FileTask(
                    task_id="task-1",
                    relative_path="README.md",
                    action="create",
                    description="Create an initial project README from the user prompt.",
                )
            ],
        )

    @staticmethod
    def _extract_executor_output(raw_state: dict[str, Any]) -> ExecutorOutput:
        text = _extract_agent_text(raw_state)
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    status = "success" if parsed.get("status") == "success" or parsed.get("exit_code") == 0 else "error"
                    result = ExecutorCommandResult(
                        command_phase="test",
                        command=[],
                        status=status,
                        exit_code=parsed.get("exit_code"),
                        stdout=str(parsed.get("stdout", "")),
                        stderr=str(parsed.get("stderr", "")),
                        message=str(parsed.get("message", "")),
                    )
                    return ExecutorOutput(
                        status=status,
                        command_results=[result],
                        summary=text,
                        raw_agent_state=raw_state,
                    )
            except json.JSONDecodeError:
                pass

        lowered = (text or "").lower()
        status = "success" if "success" in lowered and "error" not in lowered else "error"
        return ExecutorOutput(
            status=status,
            summary=text or "Executor did not return command output.",
            raw_agent_state=raw_state,
        )

    @staticmethod
    def _file_tasks_or_default(supervisor_output: SupervisorOutput) -> list[FileTask]:
        if supervisor_output.file_tasks:
            return supervisor_output.file_tasks
        return [
            FileTask(
                task_id="task-1",
                relative_path="README.md",
                action="create",
                description="Create an initial project README from the supervisor plan.",
            )
        ]

    @staticmethod
    def _build_task_store(file_tasks: list[FileTask]) -> list[FileTaskProgress]:
        return [
            FileTaskProgress(
                task_id=task.task_id,
                relative_path=task.relative_path,
                status="pending",
            )
            for task in file_tasks
        ]

    @staticmethod
    def _next_pending_task_index(task_store: list[FileTaskProgress]) -> int | None:
        for index, task in enumerate(task_store):
            if task.status in {"pending", "failed"}:
                return index
        return None

    @staticmethod
    def _update_task_status(
        task_store: list[FileTaskProgress],
        task_id: str,
        status: Literal["pending", "written_untested", "tested", "failed"],
    ) -> list[FileTaskProgress]:
        return [
            task.model_copy(update={"status": status}) if task.task_id == task_id else task
            for task in task_store
        ]

    @staticmethod
    def _mark_written_tasks_tested(task_store: list[FileTaskProgress]) -> list[FileTaskProgress]:
        return [
            task.model_copy(update={"status": "tested"}) if task.status == "written_untested" else task
            for task in task_store
        ]

    @staticmethod
    def _reset_written_tasks_for_repair(task_store: list[FileTaskProgress]) -> list[FileTaskProgress]:
        return [
            task.model_copy(update={"status": "failed"}) if task.status == "written_untested" else task
            for task in task_store
        ]

    @staticmethod
    def _executor_commands_or_summary(executor_output: ExecutorOutput) -> list[str]:
        commands = [
            " ".join(result.command)
            for result in executor_output.command_results
            if result.command
        ]
        if commands:
            return commands
        return [executor_output.summary] if executor_output.summary else []

    @staticmethod
    def _append_trace(
        state: MediumAgentState,
        agent: str,
        step: str,
        step_input: dict[str, Any],
        step_output: dict[str, Any],
        approval: ApprovalDecision,
    ) -> list[TraceEvent]:
        return [
            *state.get("trace", []),
            TraceEvent(
                agent=agent,
                step=step,
                input=step_input,
                output=step_output,
                approval=approval,
            ),
        ]

    @staticmethod
    def _increment_revision(state: MediumAgentState, agent: str) -> MediumAgentState:
        revision_attempts = dict(state.get("revision_attempts", {}))
        revision_attempts[agent] = revision_attempts.get(agent, 0) + 1
        return {**state, "revision_attempts": revision_attempts}

    @staticmethod
    def _revision_limit_reached(state: MediumAgentState, agent: str) -> bool:
        return state.get("revision_attempts", {}).get(agent, 0) >= state.get("max_revision_attempts", 3)

    @staticmethod
    def _latest_rejection_message(state: MediumAgentState, agent: str) -> str:
        for event in reversed(state.get("trace", [])):
            if event.agent == agent and event.approval and not event.approval.approved:
                return event.approval.message
        return ""

    def _announce_agent(self, agent_name: str, state: MediumAgentState) -> None:
        if self.progress_callback is not None:
            self.progress_callback(agent_name, {**state, "current_agent": agent_name})


class ExecutorWriterGraph(AgentSystemMediumGraph):
    """Compatibility name for the medium graph."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("graph_topology", "writer_executor")
        super().__init__(*args, **kwargs)

    def run(
        self,
        user_task: str,
        testing_framework: str = "pytest",
        max_attempts: int = 5,
    ) -> dict[str, Any]:
        return super().run(
            user_prompt=f"{user_task}\n\nTesting framework: {testing_framework}",
            max_repair_attempts=max_attempts,
        )


def _runner_error_message(agent_name: str, error: Exception) -> str:
    return f"{agent_name} runner failed with {type(error).__name__}: {error}"


def _extract_agent_text(raw_state: dict[str, Any]) -> str | None:
    messages = raw_state.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            content = _message_content(message)
            if content:
                return content

    for key in ("output", "result", "final_executor_output"):
        value = raw_state.get(key)
        if value is not None:
            return str(value)
    return None


def _message_content(message: Any) -> str | None:
    if isinstance(message, ToolMessage):
        return str(message.content)
    if isinstance(message, BaseMessage):
        return str(message.content)
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message["content"])
    return None


if __name__ == "__main__":
    print("Agent System Medium")
    user_prompt = input("Enter your project prompt: ").strip()
    if not user_prompt:
        raise SystemExit("Prompt is required.")

    approval_mode = input("Auto-approve agent steps? [Y/n]: ").strip().lower()
    approval_callback = interactive_approve if approval_mode in {"n", "no"} else auto_approve

    topology = input("Graph topology [supervisor_centered/writer_executor]: ").strip().lower()
    if topology not in {"", "supervisor_centered", "writer_executor"}:
        raise SystemExit("Topology must be supervisor_centered or writer_executor.")

    graph = AgentSystemMediumGraph(
        approval_callback=approval_callback,
        graph_topology=topology or "supervisor_centered",
    )
    result = graph.run(user_prompt)

    print("\nFinal result:")
    print(json.dumps(result, indent=2, default=str))

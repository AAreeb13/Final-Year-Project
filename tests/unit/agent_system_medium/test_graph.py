from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage

from src.agent_system_medium.graph import AgentSystemMediumGraph, auto_approve
from src.agent_system_medium.schemas import ApprovalDecision, FileTask, SupervisorOutput
from src.multi_agent_system.executor_writer import ExecutorWriterGraph


class FakeAgent:
    pass


def _supervisor_state(file_tasks: list[FileTask] | None = None) -> dict[str, Any]:
    output = SupervisorOutput(
        project_summary="Build a tiny project.",
        functional_requirements=["Feature works"],
        architecture_design=["Keep modules separate"],
        repository_structure=["src/app.py", "tests/test_app.py"],
        file_tasks=file_tasks
        or [
            FileTask(
                task_id="app",
                relative_path="src/app.py",
                description="Create app implementation.",
            )
        ],
        dependency_plan={"setup_commands": [["python", "-m", "pip", "install", "-r", "requirements.txt"]]},
        test_plan={"testing_framework": "pytest", "test_commands": [["pytest", "-q"]]},
    )
    return {"structured_response": output}


def _executor_state(status: str = "success", exit_code: int = 0, stderr: str = "") -> dict[str, Any]:
    return {
        "messages": [
            ToolMessage(
                content=json.dumps(
                    {
                        "status": status,
                        "exit_code": exit_code,
                        "stdout": "ok\n" if status == "success" else "",
                        "stderr": stderr,
                        "message": status,
                    }
                ),
                name="run_in_existing_container",
                tool_call_id="call-1",
            )
        ]
    }


def test_legacy_executor_writer_import_points_to_medium_system() -> None:
    assert issubclass(ExecutorWriterGraph, AgentSystemMediumGraph)


def test_named_constructors_select_graph_topology() -> None:
    supervisor_centered = AgentSystemMediumGraph.supervisor_centered(
        supervisor_agent=FakeAgent(),
        writer_agent=FakeAgent(),
        executor_agent=FakeAgent(),
    )
    writer_executor = AgentSystemMediumGraph.writer_executor(
        supervisor_agent=FakeAgent(),
        writer_agent=FakeAgent(),
        executor_agent=FakeAgent(),
    )

    assert supervisor_centered.graph_topology == "supervisor_centered"
    assert writer_executor.graph_topology == "writer_executor"


def test_supervisor_output_accepts_string_commands() -> None:
    output = SupervisorOutput.model_validate(
        {
            "dependency_plan": {"setup_commands": ["pip install unittest"]},
            "test_plan": {"test_commands": ["python -m unittest test_calculator.py"]},
        }
    )

    assert output.dependency_plan.setup_commands == [["pip", "install", "unittest"]]
    assert output.test_plan.test_commands == [["python", "-m", "unittest", "test_calculator.py"]]


def test_supervisor_output_accepts_command_objects() -> None:
    output = SupervisorOutput.model_validate(
        {
            "dependency_plan": {
                "setup_commands": [
                    {"command": "python -m pip install pytest", "reason": "Install the test runner."}
                ]
            },
            "test_plan": {
                "test_commands": [
                    {"command": "pytest tests/test_game.py", "description": "Validate game logic."}
                ]
            },
        }
    )

    assert output.dependency_plan.setup_commands == [["python", "-m", "pip", "install", "pytest"]]
    assert output.test_plan.test_commands == [["pytest", "tests/test_game.py"]]


def test_medium_graph_routes_supervisor_to_writer_then_executor() -> None:
    supervisor_prompts: list[str] = []
    writer_prompts: list[str] = []
    executor_prompts: list[str] = []
    running_agents: list[str] = []

    graph = AgentSystemMediumGraph(
        supervisor_agent=FakeAgent(),
        writer_agent=FakeAgent(),
        executor_agent=FakeAgent(),
        supervisor_runner=lambda agent, prompt, config: supervisor_prompts.append(prompt) or _supervisor_state(),
        writer_runner=lambda agent, prompt, config: writer_prompts.append(prompt) or {"messages": [{"content": "wrote file"}]},
        executor_runner=lambda agent, prompt, config: executor_prompts.append(prompt) or _executor_state(),
        approval_callback=auto_approve,
        progress_callback=lambda agent_name, state: running_agents.append(agent_name),
    )

    result = graph.run("Build a tiny project")

    assert result["status"] == "success"
    assert len(result["writer_outputs"]) == 1
    assert result["writer_outputs"][0]["relative_path"] == "src/app.py"
    assert "setup_commands" in executor_prompts[0]
    assert '"stage": "start"' in supervisor_prompts[0]
    assert '"stage": "coding"' in supervisor_prompts[1]
    assert '"stage": "execution"' in supervisor_prompts[2]
    assert result["progress_update"]["stage"] == "execution"
    assert result["progress_update"]["status"] == "complete"
    assert result["task_store"] == [{"task_id": "app", "relative_path": "src/app.py", "status": "tested"}]
    assert [event["agent"] for event in result["trace"]] == [
        "supervisor",
        "writer",
        "supervisor",
        "executor",
        "supervisor",
    ]
    assert running_agents == ["supervisor", "writer", "supervisor", "executor", "supervisor"]
    assert result["current_agent"] == "supervisor"


def test_writer_processes_file_tasks_sequentially() -> None:
    file_tasks = [
        FileTask(task_id="app", relative_path="src/app.py", description="Create implementation."),
        FileTask(task_id="test", relative_path="tests/test_app.py", description="Create tests."),
    ]
    writer_prompts: list[str] = []
    executor_prompts: list[str] = []

    graph = AgentSystemMediumGraph(
        supervisor_agent=FakeAgent(),
        writer_agent=FakeAgent(),
        executor_agent=FakeAgent(),
        supervisor_runner=lambda agent, prompt, config: _supervisor_state(file_tasks),
        writer_runner=lambda agent, prompt, config: writer_prompts.append(prompt) or {"messages": [{"content": "done"}]},
        executor_runner=lambda agent, prompt, config: executor_prompts.append(prompt) or _executor_state(),
        approval_callback=auto_approve,
    )

    result = graph.run("Build a tiny project")

    assert result["status"] == "success"
    assert [output["task_id"] for output in result["writer_outputs"]] == ["app", "test"]
    assert "src/app.py" in writer_prompts[0]
    assert "tests/test_app.py" in writer_prompts[1]
    assert len(executor_prompts) == 1
    assert [task["status"] for task in result["task_store"]] == ["tested", "tested"]


def test_executor_failure_routes_back_to_writer_repair() -> None:
    executor_calls = 0
    writer_prompts: list[str] = []

    def executor_runner(agent: Any, prompt: str, config: dict[str, Any]) -> dict[str, Any]:
        nonlocal executor_calls
        executor_calls += 1
        if executor_calls == 1:
            return _executor_state("error", 1, "AssertionError")
        return _executor_state()

    graph = AgentSystemMediumGraph(
        supervisor_agent=FakeAgent(),
        writer_agent=FakeAgent(),
        executor_agent=FakeAgent(),
        supervisor_runner=lambda agent, prompt, config: _supervisor_state(),
        writer_runner=lambda agent, prompt, config: writer_prompts.append(prompt) or {"messages": [{"content": "updated"}]},
        executor_runner=executor_runner,
        approval_callback=auto_approve,
    )

    result = graph.run("Repair after failure")

    assert result["status"] == "success"
    assert result["attempts"] == 1
    assert len(writer_prompts) == 2
    assert "AssertionError" in writer_prompts[1]
    assert [event["agent"] for event in result["trace"]] == [
        "supervisor",
        "writer",
        "supervisor",
        "executor",
        "supervisor",
        "writer",
        "supervisor",
        "executor",
        "supervisor",
    ]


def test_human_rejection_routes_back_to_relevant_agent() -> None:
    supervisor_calls = 0

    def supervisor_runner(agent: Any, prompt: str, config: dict[str, Any]) -> dict[str, Any]:
        nonlocal supervisor_calls
        supervisor_calls += 1
        assert supervisor_calls == 1 or "Add clearer requirements" in prompt
        return _supervisor_state()

    def approval(agent_name: str, step_input: dict[str, Any], step_output: dict[str, Any]) -> ApprovalDecision:
        if agent_name == "supervisor" and supervisor_calls == 1:
            return ApprovalDecision(approved=False, message="Add clearer requirements")
        return ApprovalDecision(approved=True)

    graph = AgentSystemMediumGraph(
        supervisor_agent=FakeAgent(),
        writer_agent=FakeAgent(),
        executor_agent=FakeAgent(),
        supervisor_runner=supervisor_runner,
        writer_runner=lambda agent, prompt, config: {"messages": [{"content": "done"}]},
        executor_runner=lambda agent, prompt, config: _executor_state(),
        approval_callback=approval,
    )

    result = graph.run("Build a tiny project")

    assert result["status"] == "success"
    assert supervisor_calls == 4
    assert result["trace"][0]["approval"]["approved"] is False
    assert result["trace"][1]["approval"]["approved"] is True

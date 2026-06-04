import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

from src.evaluation.system_eval_schema import SystemRunOutput
from src.tools.python_code_execution.tool import run_python_code_tool

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver


def import_settings():
    from src.settings import settings
    return settings


PROMPTS_DIR = Path(__file__).with_name("system_prompts")


def available_prompt_names() -> list[str]:
    return sorted(prompt_path.stem for prompt_path in PROMPTS_DIR.glob("*.txt"))


def load_system_prompt(prompt_name: str) -> str:
    prompt_path = PROMPTS_DIR / f"{prompt_name}.txt"

    if not prompt_path.exists():
        available = ", ".join(available_prompt_names())
        raise ValueError(
            f"Unknown prompt '{prompt_name}'. Available prompts: {available}"
        )

    return prompt_path.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single-agent SDLC workflow for a given problem statement."
    )

    parser.add_argument(
        "problem_statement",
        help="Natural-language problem statement for the agent to solve.",
    )

    parser.add_argument(
        "--prompt",
        default="master",
        choices=available_prompt_names(),
        help="Which system prompt to use for this run.",
    )

    parser.add_argument(
        "--project-id",
        default=None,
        help="Optional project ID used by the evaluation harness.",
    )

    return parser.parse_args()

def build_agent(settings, prompt_name: str):
    assert settings.OPEN_ROUTER_KEY is not None, (
        "OPEN_ROUTER_KEY must be set in the .env file"
    )
    system_prompt = load_system_prompt(prompt_name)

    model = ChatOpenAI(
        api_key=SecretStr(settings.OPEN_ROUTER_KEY),
        base_url="https://openrouter.ai/api/v1",
        model="gpt-4o-mini",
        temperature=0.1,
        max_completion_tokens=100000,
)

    tools = [run_python_code_tool]

    interrupt_on = {
        tool.name: {"allowed_decisions": ["approve", "edit", "reject"]}
        for tool in tools
    }
    agent = create_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        response_format=SystemRunOutput,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on=interrupt_on,
                description_prefix="Tool call pending human approval",
            )
        ],
        checkpointer=InMemorySaver(),
    )

    return agent

def normalise_output(
    response: SystemRunOutput,
    project_id: str | None = None,
) -> SystemRunOutput:
    """
    Ensures every single-agent run has the required evaluation metadata.
    """

    if project_id is not None:
        response.project_id = project_id

    if not response.system_name:
        response.system_name = "single_agent"

    if response.status == "unknown":
        response.status = "completed"

    return response


def execute(
    problem_statement: str,
    prompt_name: str = "master",
    project_id: str | None = None,
) -> tuple[SystemRunOutput, str]:
    settings = import_settings()
    agent = build_agent(settings, prompt_name)

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=problem_statement)
            ]
        }
    )

    structured_response = result.get("structured_response")

    if isinstance(structured_response, SystemRunOutput):
        response = structured_response
    elif isinstance(structured_response, dict):
        response = SystemRunOutput.model_validate(structured_response)
    else:
        raise ValueError(
            "Agent did not return a valid structured_response. "
            "Check the model, response_format, and system prompt."
        )

    response = normalise_output(response, project_id=project_id)
    response_json = response.model_dump_json(indent=2)

    return response, response_json


def display_response(response: SystemRunOutput) -> None:
    print(f"System: {response.system_name}")
    print(f"Project ID: {response.project_id}")
    print(f"Status: {response.status}")

    print("\nFunctional Requirements:")
    for req in response.requirements.functional:
        print(f"- {req}")

    print("\nNon-Functional Requirements:")
    for req in response.requirements.non_functional:
        print(f"- {req}")

    print("\nConstraints:")
    for constraint in response.requirements.constraints:
        print(f"- {constraint}")

    print("\nHigh-Level Architecture:")
    print(f"Style: {response.high_level_design.style}")

    print("\nArchitecture Components:")
    for component in response.high_level_design.components:
        print(f"- {component.name}: {component.responsibilities}")

    print("\nComponents:")
    for component in response.components:
        print(f"- {component.name}: {component.responsibilities}")

    print("\nModules:")
    for module in response.modules:
        print(f"- {module.name} [{module.component}]: {module.responsibilities}")

    print("\nImplementation Plan:")
    for step in response.implementation_plan:
        print(f"- {step.step_id}: {step.action} -> {step.result}")

    print("\nTest Plan:")
    print(f"Framework: {response.test_plan.testing_framework}")
    for test in response.test_plan.unit_tests:
        print(f"- Unit: {test}")
    for test in response.test_plan.integration_tests:
        print(f"- Integration: {test}")

    print("\nGenerated Files:")
    for file in response.generated_files:
        print(f"- {file.path}")

    print("\nExecution Results:")
    for result in response.execution_results:
        print(f"- Command: {result.command}")
        print(f"  Success: {result.success}")
        print(f"  Exit code: {result.exit_code}")
        if result.stdout:
            print(f"  stdout: {result.stdout}")
        if result.stderr:
            print(f"  stderr: {result.stderr}")


def save_response(
    response_json: str,
    prompt_name: str,
    project_id: str | None = None,
) -> Path:
    current_directory = os.getcwd()
    response_directory = Path(current_directory) / ".response" / "single_agent"
    response_directory.mkdir(parents=True, exist_ok=True)

    cur_time = int(time.time())
    cur = hex(cur_time)[2:]

    safe_project_id = project_id or "manual"
    output_filename = f"{safe_project_id}_{prompt_name}_{cur}.json"
    output_path = response_directory / output_filename

    output_path.write_text(response_json, encoding="utf-8")

    print(f"Saved response to: {output_path}")

    return output_path


def run(
    problem_statement: str,
    prompt_name: str = "master",
    project_id: str | None = None,
) -> SystemRunOutput:
    response, response_json = execute(
        problem_statement=problem_statement,
        prompt_name=prompt_name,
        project_id=project_id,
    )

    display_response(response)
    save_response(response_json, prompt_name, project_id=project_id)

    return response


if __name__ == "__main__":
    args = parse_args()

    run(
        problem_statement=args.problem_statement,
        prompt_name=args.prompt,
        project_id=args.project_id,
    )
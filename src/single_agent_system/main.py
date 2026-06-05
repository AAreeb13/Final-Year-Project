import argparse
import os
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langsmith import tracing_context

from src.evaluation.system_eval_schema import SystemRunOutput
from src.agents.human_approval_runner import run_with_human_approval
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

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow the agent to execute generated Python validation code.",
    )

    parser.add_argument(
        "--debug-structured-output",
        action="store_true",
        help="Print the model's attempted structured response before schema validation.",
    )

    return parser.parse_args()


def configure_langsmith(settings) -> None:
    """
    Mirrors LangSmith settings from .env into os.environ for LangChain tracing.
    """

    os.environ["LANGSMITH_TRACING"] = str(settings.LANGSMITH_TRACING).lower()

    optional_env_vars = {
        "LANGSMITH_ENDPOINT": settings.LANGSMITH_ENDPOINT,
        "LANGSMITH_API_KEY": settings.LANGSMITH_API_KEY,
        "LANGSMITH_PROJECT": settings.LANGSMITH_PROJECT,
    }
    for env_var, value in optional_env_vars.items():
        if value:
            os.environ[env_var] = value


def build_agent(settings, prompt_name: str, allow_tool_execution: bool = False):
    assert settings.OPEN_ROUTER_KEY is not None, (
        "OPEN_ROUTER_KEY must be set in the .env file"
    )
    system_prompt = load_system_prompt(prompt_name)

    model = ChatOpenAI(
        api_key=SecretStr(settings.OPEN_ROUTER_KEY),
        base_url="https://openrouter.ai/api/v1",
        model="gpt-4o-mini",
        temperature=0.1,
        # max_completion_tokens=100000,
)

    tools = [run_python_code_tool] if allow_tool_execution else []

    interrupt_on = {
        tool.name: {"allowed_decisions": ["approve", "edit", "reject"]}
        for tool in tools
    }
    agent = create_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        response_format=ToolStrategy(
            SystemRunOutput,
            handle_errors=(
                "The structured response did not match the required schema. "
                "Return only fields defined by SystemRunOutput. In particular, "
                "put module-level `component` and `signatures` fields under "
                "`modules`, not under `components`."
            ),
        ),
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
    allow_tool_execution: bool = False,
    debug_structured_output: bool = False,
) -> tuple[SystemRunOutput, str]:
    settings = import_settings()
    configure_langsmith(settings)
    agent = build_agent(
        settings,
        prompt_name,
        allow_tool_execution=allow_tool_execution,
    )

    trace_tags = ["single_agent", f"prompt:{prompt_name}"]
    trace_metadata = {
        "system_name": "single_agent",
        "prompt_name": prompt_name,
    }
    if project_id is not None:
        trace_metadata["project_id"] = project_id
    thread_id = f"single-agent-{project_id or uuid.uuid4()}"
    run_config = {
        "run_name": "single_agent_system",
        "tags": trace_tags,
        "metadata": trace_metadata,
        "configurable": {
            "thread_id": thread_id,
        },
    }

    with tracing_context(
        enabled=settings.LANGSMITH_TRACING,
        project_name=settings.LANGSMITH_PROJECT,
        tags=trace_tags,
        metadata=trace_metadata,
    ):
        result = run_with_human_approval(
            agent=agent,
            user_input=problem_statement,
            config=run_config,
            debug_structured_output=debug_structured_output,
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

    if response.requirements.functional:
        print("\nFunctional Requirements:")
        for req in response.requirements.functional:
            print(f"- {req}")

    if response.requirements.non_functional:
        print("\nNon-Functional Requirements:")
        for req in response.requirements.non_functional:
            print(f"- {req}")

    if response.requirements.constraints:
        print("\nConstraints:")
        for constraint in response.requirements.constraints:
            print(f"- {constraint}")

    print("\nHigh-Level Architecture:")
    print(f"Style: {response.high_level_design.style}")

    if response.high_level_design.components:
        print("\nArchitecture Components:")
        for component in response.high_level_design.components:
            print(f"- {component.name}: {component.responsibilities}")

    if response.components:
        print("\nComponents:")
        for component in response.components:
            print(f"- {component.name}: {component.responsibilities}")

    if response.modules:
        print("\nModules:")
        for module in response.modules:
            print(f"- {module.name} [{module.component}]: {module.responsibilities}")

    if response.implementation_plan:
        print("\nImplementation Plan:")
        for step in response.implementation_plan:
            print(f"- {step.step_id}: {step.action} -> {step.result}")

    if (
        response.test_plan.testing_framework
        or response.test_plan.unit_tests
        or response.test_plan.integration_tests
    ):
        print("\nTest Plan:")
        if response.test_plan.testing_framework:
            print(f"Framework: {response.test_plan.testing_framework}")
        for test in response.test_plan.unit_tests:
            print(f"- Unit: {test}")
        for test in response.test_plan.integration_tests:
            print(f"- Integration: {test}")

    if response.generated_files:
        print("\nGenerated Files:")
        for file in response.generated_files:
            print(f"- {file.path}")

    if response.execution_results:
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
    allow_tool_execution: bool = False,
    debug_structured_output: bool = False,
) -> SystemRunOutput:
    response, response_json = execute(
        problem_statement=problem_statement,
        prompt_name=prompt_name,
        project_id=project_id,
        allow_tool_execution=allow_tool_execution,
        debug_structured_output=debug_structured_output,
    )

    display_response(response)
    if not project_id:
        project_id = input("Enter a project ID to associate with this run (or press Enter to skip): ").strip() or ""
        project_id = project_id + "_manual" + "_" + hex(int(time.time()))[2:]
    save_response(response_json, prompt_name, project_id=project_id)

    return response


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_args()
        run(
            problem_statement=args.problem_statement,
            prompt_name=args.prompt,
            project_id=args.project_id,
            allow_tool_execution=args.execute,
            debug_structured_output=args.debug_structured_output,
        )
    else:
        problem_statement = input("Enter the problem statement for the agent to solve: ").strip()
        run(
            problem_statement=problem_statement,
            prompt_name="system_design",
            project_id=None,
            allow_tool_execution=False,
            debug_structured_output=True,
        )

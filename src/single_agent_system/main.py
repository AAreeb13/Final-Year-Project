
import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, SecretStr

from src.tools.python_code_execution.tool import run_python_code_tool

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
        raise ValueError(f"Unknown prompt '{prompt_name}'. Available prompts: {available}")
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
    return parser.parse_args()


class SingleAgentModelOutput(BaseModel):
    # Task Decomposition
    functional_requirements: List[str] = Field(default_factory=list)
    # Top-Down Decomposition

    components: List[str] = Field(default_factory=list)
    relationships: List[str] = Field(default_factory=list)
    # System Design
    system_design: List[str] = Field(default_factory=list)
    solid_principles_used: List[str] = Field(default_factory=list)
    solid_principles_jeopardised: List[str] = Field(default_factory=list)

    folder_structure_json: str = ""
    # Code Generation
    filename_to_code: dict[str, str] = Field(default_factory=dict)  # filename: code

    files_run_with_output: dict[str, str] = Field(default_factory=dict)  # filename: output


def build_agent(settings, prompt_name: str):
    assert settings.OPEN_ROUTER_KEY is not None, "openrouterkey must be set in the .env file"
    print("Settings imported successfully. openrouterkey is set.")
    system_prompt = load_system_prompt(prompt_name)
    print(f"Using system prompt: {prompt_name}")
    model = ChatOpenAI(
        api_key=SecretStr(settings.OPEN_ROUTER_KEY),
        base_url="https://openrouter.ai/api/v1",
        model="gpt-4o-mini",
        temperature=0.1,
        max_completion_tokens=100000)
    
    
    
    agent = create_agent(
        model=model,
        system_prompt=system_prompt,
        tools=[run_python_code_tool],
        response_format=SingleAgentModelOutput
    )
    return agent


def display_response(response: SingleAgentModelOutput) -> None:
    print("Functional Requirements:")
    for req in response.functional_requirements:
        print(f"- {req}")
    print("\nComponents:")
    for comp in response.components:
        print(f"- {comp}")
    print("\nRelationships:")
    for rel in response.relationships:
        print(f"- {rel}")
    print("\nSystem Design:")
    for design_item in response.system_design:
        print(f"- {design_item}")
    print("\nSOLID Principles Used:")
    for solid in response.solid_principles_used:
        print(f"- {solid}")
    print("\nSOLID Principles Jeopardised:")
    for solid in response.solid_principles_jeopardised:
        print(f"- {solid}")
    print("\nFolder Structure:")
    print(response.folder_structure_json)
    print("\nGenerated Code:")
    for filename, code in response.filename_to_code.items():
        print(f"Filename: {filename}\nCode:\n{code}\n{'-'*40}")
    print("\nCode Execution Output:") 
    for filename, output in response.files_run_with_output.items():
        print(f"Filename: {filename}\nOutput:\n{output}\n{'-'*40}")


def save_response(response_json: str, prompt_name: str) -> None:
    current_directory = os.getcwd()
    response_directory = os.path.join(current_directory, ".response")
    os.makedirs(response_directory, exist_ok=True)
    cur_time = int(time.time())
    cur = hex(cur_time)[2:]
    output_filename = f"{prompt_name}_response{cur}.json"
    with open(os.path.join(response_directory, output_filename), "w") as f:
        print("Saving response to file at location:", os.path.join(response_directory, output_filename))
        f.write(response_json)


def run(problem_statement: str, prompt_name: str = "master") -> SingleAgentModelOutput:
    settings = import_settings()
    agent = build_agent(settings, prompt_name)
    response_dict = agent.invoke({"messages": [HumanMessage(problem_statement)]})
    response_json = response_dict["messages"][-1].content
    response = SingleAgentModelOutput.model_validate_json(response_json)
    display_response(response)
    save_response(response_json, prompt_name)
    return response


if __name__ == "__main__":
    args = parse_args()
    run(args.problem_statement, args.prompt)
        
 

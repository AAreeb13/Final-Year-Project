
import uuid

from src.tools.docker_code_execution.repo_tool import run_repository_command
from src.tools.docker_code_execution.tool import stop_container

from src.agents.AgentFactory import AgentFactory
from src.agents.human_approval_runner import run_with_human_approval
from src.tools.file_system_tools.inspect_repository import (
    inspect_repository,
    inspect_repository_file,
)


def load_code_executor_system_prompt() -> str:
    with open("src/agents/code_executor/system_prompt.txt", "r") as f:
        return f.read()

def build_code_executor_agent():
    prompt = load_code_executor_system_prompt()
    tools = [
        inspect_repository,
        inspect_repository_file,
        run_repository_command,
        stop_container,
    ]
    return AgentFactory.build_agent_with_HITLmiddleware_InMemCheckpointer(
        prompt=prompt,
        tools=tools,
    )

if __name__ == "__main__":
    agent = build_code_executor_agent()
    session_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    while True:
        user_input = input("Enter your command (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break

        run_with_human_approval(agent, user_input, session_config)

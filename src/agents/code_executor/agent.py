
from src.agents.AgentFactory import AgentFactory
from src.tools.file_system_tools.inspect_file import inspect_file
from src.tools.file_system_tools.inspect_workplace import inspect_folder_structure


def build_code_executor_agent():
    prompt = load_code_executor_system_prompt()
    tools = [
        inspect_folder_structure,
        inspect_file,
    ]
    return AgentFactory.build_agent(
        prompt=prompt,
        tools=tools,
    )
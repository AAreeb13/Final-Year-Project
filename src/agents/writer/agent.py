from tools.file_system_tools.edit_file import edit_file
from tools.file_system_tools.inspect_file import inspect_file
from tools.file_system_tools.inspect_workplace import inspect_folder_structure

from src.agents.AgentFactory import AgentFactory




def build_writer_agent():
    prompt = load_writer_system_prompt()
    tools = [
        inspect_folder_structure,
        inspect_file,
        edit_file,
        ]
    return AgentFactory.build_agent(prompt=prompt)

def load_writer_system_prompt():
    with open("src/agents/writer/system_prompt.txt", "r") as f:
        return f.read()
    

if __name__ == "__main__":
    print(load_writer_system_prompt())
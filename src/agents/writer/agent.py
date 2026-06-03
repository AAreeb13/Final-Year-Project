import uuid

from src.tools.file_system_tools.manage_file import delete_file, edit_file, create_file
from src.tools.file_system_tools.inspect_file import inspect_file
from src.tools.file_system_tools.inspect_workplace import inspect_folder_structure
from src.tools.file_system_tools.manage_folders import create_folder, delete_folder

from src.agents.AgentFactory import AgentFactory
from src.agents.human_approval_runner import run_with_human_approval

def build_writer_agent():
    prompt = load_writer_system_prompt()
    tools = [
        inspect_folder_structure,
        inspect_file,
        create_file,
        edit_file,
        delete_file,
        create_folder,
        delete_folder,
    ]
    return AgentFactory.build_agent_with_HITLmiddleware_InMemCheckpointer(
        prompt=prompt,
        tools=tools,
    )


def load_writer_system_prompt():
    with open("src/agents/writer/system_prompt.txt", "r") as f:
        return f.read()
    

if __name__ == "__main__":

    agent = build_writer_agent()
    print("Agent created successfully.")
    session_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # create a loop for us to interact with the agent in the console and ask for permission to use tools

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        
        run_with_human_approval(agent, user_input, session_config)




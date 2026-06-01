import json
import uuid

from tools.file_system_tools.manage_file import edit_file, create_file
from tools.file_system_tools.inspect_file import inspect_file
from tools.file_system_tools.inspect_workplace import inspect_folder_structure
from tools.file_system_tools.manage_folders import create_folder, delete_folder

from langchain_core.messages import HumanMessage
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.agents.AgentFactory import AgentFactory




def build_writer_agent():
    prompt = load_writer_system_prompt()
    tools = [
        inspect_folder_structure,
        inspect_file,
        create_file,
        edit_file,
        create_folder,
        delete_folder,
    ]
    middleware = [
        HumanInTheLoopMiddleware(
            interrupt_on={
                tool.name: {"allowed_decisions": ["approve", "reject"]}
                for tool in tools
            }
        )
    ]
    return AgentFactory.build_agent(
        prompt=prompt,
        tools=tools,
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )


def _print_interrupt_requests(interrupt_payload) -> None:
    for request in interrupt_payload["action_requests"]:
        print("\nApproval required")
        print(f"Tool: {request['name']}")
        print(f"Args: {json.dumps(request['args'], indent=2)}")
        if description := request.get("description"):
            print(description)


def _collect_decisions(interrupt_payload) -> dict:
    decisions = []
    for request in interrupt_payload["action_requests"]:
        print(f"Approve tool call for {request['name']}? Args: {json.dumps(request['args'], indent=2)}")
        while True:
            choice = input("Approve this tool call? [y/n]: ").strip().lower()
            if choice in {"y", "yes"}:
                decisions.append({"type": "approve"})
                break
            if choice in {"n", "no"}:
                message = input("Reason for rejection (optional): ").strip()
                decision = {"type": "reject"}
                if message:
                    decision["message"] = message
                decisions.append(decision)
                break
            print("Please answer y or n.")

    return {"decisions": decisions}


def _run_with_human_approval(agent, user_input: str, config: dict) -> dict:
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

    while "__interrupt__" in response:
        resume_payload = {}
        for interrupt_item in response["__interrupt__"]:
            _print_interrupt_requests(interrupt_item.value)
            resume_payload[interrupt_item.id] = _collect_decisions(interrupt_item.value)

        response = agent.invoke(Command(resume=resume_payload), config=config)
    # After interrupts are resolved, print messages and return
    for m in response.get("messages", []):
        mt = m.__class__.__name__ if hasattr(m, "__class__") else str(type(m))
        content = getattr(m, "content", None)
        print(f"[{mt}] {content}")

    return response

def load_writer_system_prompt():
    with open("src/agents/writer/system_prompt.txt", "r") as f:
        return f.read()
    

if __name__ == "__main__":
    print("Writer prompt:")
    print("==================")
    for i in range(5):
        print(load_writer_system_prompt().strip().splitlines()[i])
    # print(load_writer_system_prompt().strip().splitlines()[i] for i in range(5))
    print("==================")

    agent = build_writer_agent()
    print("Agent created successfully.")
    session_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # create a loop for us to interact with the agent in the console and ask for permission to use tools

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        
        response = _run_with_human_approval(agent, user_input, session_config)
        # Print each returned message (SystemMessage, HumanMessage, ToolMessage, etc.)
        for m in response.get("messages", []):
            mt = m.__class__.__name__ if hasattr(m, "__class__") else str(type(m))
            content = getattr(m, "content", None)
            print(f"[{mt}] {content}")




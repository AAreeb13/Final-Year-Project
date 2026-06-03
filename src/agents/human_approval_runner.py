import json

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command


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


def _print_tool_outputs_from_chunk(chunk) -> None:
    for node_update in chunk.values():
        if not isinstance(node_update, dict):
            continue

        for message in node_update.get("messages", []):
            if isinstance(message, ToolMessage):
                print("\nTool executed")
                print(f"Tool: {message.name}")
                print("Output:")
                print(message.content)


def _extract_interrupts_from_chunk(chunk):
    if "__interrupt__" in chunk:
        return chunk["__interrupt__"]

    for node_update in chunk.values():
        if isinstance(node_update, dict) and "__interrupt__" in node_update:
            return node_update["__interrupt__"]

    return None


def run_with_human_approval(agent, user_input: str, config: dict) -> dict:
    stream_input = {"messages": [HumanMessage(content=user_input)]}

    while True:
        interrupted = False

        for chunk in agent.stream(stream_input, config=config, stream_mode="updates"):
            interrupts = _extract_interrupts_from_chunk(chunk)
            if interrupts:
                resume_payload = {}
                for interrupt_item in interrupts:
                    _print_interrupt_requests(interrupt_item.value)
                    resume_payload[interrupt_item.id] = _collect_decisions(interrupt_item.value)

                stream_input = Command(resume=resume_payload)
                interrupted = True
                break

            _print_tool_outputs_from_chunk(chunk)

        if not interrupted:
            break

    return agent.get_state(config).values

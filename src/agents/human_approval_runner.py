import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command


STRUCTURED_OUTPUT_TOOL_NAMES = {"SystemRunOutput"}


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


def _print_structured_output_attempts_from_chunk(chunk) -> None:
    for node_update in chunk.values():
        if not isinstance(node_update, dict):
            continue

        for message in node_update.get("messages", []):
            if not isinstance(message, AIMessage):
                continue

            for tool_call in getattr(message, "tool_calls", []) or []:
                if tool_call.get("name") not in STRUCTURED_OUTPUT_TOOL_NAMES:
                    continue

                print("\nStructured output attempted")
                print(f"Schema: {tool_call['name']}")
                print("Args:")
                print(json.dumps(tool_call.get("args", {}), indent=2))


def _print_tool_outputs_from_chunk(chunk) -> None:
    for node_update in chunk.values():
        if not isinstance(node_update, dict):
            continue

        for message in node_update.get("messages", []):
            if isinstance(message, ToolMessage):
                if message.name in STRUCTURED_OUTPUT_TOOL_NAMES:
                    print("\nStructured output validation")
                    print(f"Schema: {message.name}")
                    if str(message.content).startswith("Returning structured response:"):
                        print("Result: accepted")
                    else:
                        print("Result:")
                        print(message.content)
                else:
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


def run_with_human_approval(
    agent,
    user_input: str,
    config: dict,
    debug_structured_output: bool = False,
) -> dict:
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

            if debug_structured_output:
                _print_structured_output_attempts_from_chunk(chunk)
            _print_tool_outputs_from_chunk(chunk)

        if not interrupted:
            break

    return agent.get_state(config).values

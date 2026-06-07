from __future__ import annotations

from src.agent_system_medium.schemas import SupervisorOutput
from src.agents.AgentFactory import AgentFactory


def load_supervisor_system_prompt() -> str:
    with open("src/agents/supervisor/system_prompt.txt", "r", encoding="utf-8") as prompt_file:
        return prompt_file.read()


def build_supervisor_agent():
    return AgentFactory.build_agent_with_HITLmiddleware_InMemCheckpointer(
        prompt=load_supervisor_system_prompt(),
        tools=[],
        temperature=0.2,
        response_format=SupervisorOutput,
    )

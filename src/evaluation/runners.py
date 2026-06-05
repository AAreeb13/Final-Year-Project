from typing import Any

from src.evaluation.AgentSystem import AgentSystemRunner
from src.single_agent_system.main import execute as single_agent_execute

class SingleAgentSystemRunner(AgentSystemRunner):
    def __init__(self,  description: str):
        self.system_id = "single_agent_system"
        self.description = description
        self.system_prompt = "master"
        print("====== Initializing SingleAgentSystemRunner ======")
        print(f"====== Description: {description} \n====== System prompt: {self.system_prompt}")
        print("==================================================")

    def run(self, prompt: str, run_config: dict[str, Any]) -> tuple[dict[str, Any], str]:
        response, _response_json = single_agent_execute(
            problem_statement=prompt,
            prompt_name=self.system_prompt,
            allow_tool_execution=run_config.get("allow_tool_execution", False),
            debug_structured_output=run_config.get("debug_structured_output", False),
        )
        return response, f"run_{self.system_id}"

    def display_architecture(self) -> None:
        print(f"SingleAgentSystem using {self.system_prompt}") 

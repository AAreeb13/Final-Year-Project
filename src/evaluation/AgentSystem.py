from abc import abstractmethod
from typing import Any


class AgentSystemRunner:
    system_id: str
    description: str

    @abstractmethod
    def run(self, prompt: str, run_config: dict[str, Any]) -> tuple[dict[str, Any], str]:
        pass

    @abstractmethod
    def display_architecture(self) -> None:
        pass



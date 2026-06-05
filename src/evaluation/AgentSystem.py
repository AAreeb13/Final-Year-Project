from abc import abstractmethod
from typing import Any


class AgentSystem:
    system_id: str
    description: str

    @abstractmethod
    def run(self, prompt_id) -> tuple[dict[str, Any], str]:
        pass

    @abstractmethod
    def display_architecture(self) -> None:
        pass

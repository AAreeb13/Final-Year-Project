from abc import abstractmethod
from typing import Any


class AgentSystemRunner:
    system_id: str
    description: str

    def reset_system(self, *, reason: str = "", verbose: bool = False) -> None:
        """Clear runtime state before or after an evaluation run.

        Most runners are naturally stateless because they create their graph
        objects inside run(). Stateful runners can override this hook to stop
        containers, clear checkpointers, or drop cached agents.
        """
        return None

    @abstractmethod
    def run(self, prompt: str, run_config: dict[str, Any]) -> tuple[dict[str, Any], str]:
        pass

    @abstractmethod
    def display_architecture(self) -> None:
        pass



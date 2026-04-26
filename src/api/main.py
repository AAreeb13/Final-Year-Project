from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from flask import Flask, jsonify, request

from src.single_agent_system.main import available_prompt_names
from src.single_agent_system.main import run as run_single_agent


class AgentSystem(ABC):
	@property
	@abstractmethod
	def system_id(self) -> str:
		pass

	@property
	def description(self) -> str:
		return ""

	@property
	def default_prompt(self) -> str | None:
		return None

	def list_prompts(self) -> list[str]:
		return []

	@abstractmethod
	def run(self, payload: dict[str, Any]) -> dict[str, Any]:
		pass


@dataclass(slots=True)
class SingleAgentSystem(AgentSystem):
	@property
	def system_id(self) -> str:
		return "single-agent"

	@property
	def description(self) -> str:
		return "Current single-agent SDLC workflow"

	@property
	def default_prompt(self) -> str:
		return "master"

	def list_prompts(self) -> list[str]:
		return available_prompt_names()

	def run(self, payload: dict[str, Any]) -> dict[str, Any]:
		problem_statement = payload.get("user_prompt") or payload.get("problem_statement")
		if not problem_statement:
			raise ValueError("'user_prompt' (or 'problem_statement') is required")

		prompt_name = payload.get("system_prompt") or payload.get("prompt") or self.default_prompt
		available_prompts = self.list_prompts()
		if prompt_name not in available_prompts:
			raise ValueError(
				f"Unknown prompt '{prompt_name}'. Available prompts: {', '.join(available_prompts)}"
			)

		response = run_single_agent(problem_statement=problem_statement, prompt_name=prompt_name)
		return response.model_dump()


class AgentSystemRegistry:
	def __init__(self) -> None:
		self._systems: dict[str, AgentSystem] = {}

	def register(self, system: AgentSystem) -> None:
		self._systems[system.system_id] = system

	def get(self, system_id: str) -> AgentSystem | None:
		return self._systems.get(system_id)

	def list_systems(self) -> list[dict[str, str]]:
		return [
			{
				"system_id": system.system_id,
				"description": system.description,
				"default_prompt": system.default_prompt or "",
			}
			for system in self._systems.values()
		]


def create_default_registry() -> AgentSystemRegistry:
	registry = AgentSystemRegistry()
	registry.register(SingleAgentSystem())
	return registry


def create_app(registry: AgentSystemRegistry | None = None) -> Flask:
	app = Flask(__name__)
	active_registry = registry or create_default_registry()

	@app.get("/health")
	def health() -> tuple[dict[str, str], int]:
		return {"status": "ok"}, 200

	@app.get("/systems")
	def list_systems() -> tuple[Any, int]:
		return jsonify({"systems": active_registry.list_systems()}), 200

	@app.get("/systems/<system_id>/prompts")
	def list_system_prompts(system_id: str) -> tuple[Any, int]:
		system = active_registry.get(system_id)
		if system is None:
			return jsonify({"error": f"Unknown system_id '{system_id}'"}), 404

		return jsonify(
			{
				"system_id": system_id,
				"default_prompt": system.default_prompt,
				"prompts": system.list_prompts(),
			}
		), 200

	@app.post("/systems/<system_id>/run")
	def run_system(system_id: str) -> tuple[Any, int]:
		system = active_registry.get(system_id)
		if system is None:
			return jsonify({"error": f"Unknown system_id '{system_id}'"}), 404

		payload = request.get_json(silent=True) or {}
		try:
			result = system.run(payload)
		except ValueError as exc:
			return jsonify({"error": str(exc)}), 400
		except Exception as exc:  # noqa: BLE001
			return jsonify({"error": "Execution failed", "details": str(exc)}), 500

		return jsonify({"system_id": system_id, "result": result}), 200

	return app


app = create_app()


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8000, debug=True)

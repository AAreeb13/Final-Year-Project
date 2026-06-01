from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.api.main import AgentSystem, AgentSystemRegistry, create_app

class MockAgentSystem(AgentSystem):
    @property
    def system_id(self) -> str:
        return "fake-system"

    @property
    def description(self) -> str:
        return "Fake system for API tests"

    @property
    def default_prompt(self) -> str:
        return "default"

    def list_prompts(self) -> list[str]:
        return ["default", "review"]

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_prompt = payload.get("user_prompt")
        if not user_prompt:
            raise ValueError("'user_prompt' is required")
        if user_prompt == "explode":
            raise RuntimeError("boom")
        return {"echo": user_prompt, "prompt": payload.get("prompt", self.default_prompt)}


@pytest.fixture
def client():
    registry = AgentSystemRegistry()
    registry.register(MockAgentSystem())
    app = create_app(registry)
    app.config.update(TESTING=True)

    return app.test_client()


def test_health_returns_ok(client):
    print("Testing /health endpoint...")
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_systems_lists_registered_systems(client):
    print("Testing /systems endpoint...")
    response = client.get("/systems")

    assert response.status_code == 200
    assert response.get_json() == {
        "systems": [
            {
                "system_id": "fake-system",
                "description": "Fake system for API tests",
                "default_prompt": "default",
            }
        ]
    }


def test_prompts_returns_prompts_for_known_system(client):
    response = client.get("/systems/fake-system/prompts")

    assert response.status_code == 200
    assert response.get_json() == {
        "system_id": "fake-system",
        "default_prompt": "default",
        "prompts": ["default", "review"],
    }


def test_prompts_returns_404_for_unknown_system(client):
    response = client.get("/systems/unknown/prompts")

    assert response.status_code == 404
    assert response.get_json() == {"error": "Unknown system_id 'unknown'"}


def test_run_returns_system_result(client):
    response = client.post(
        "/systems/fake-system/run",
        json={"user_prompt": "build a todo app", "prompt": "review"},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "system_id": "fake-system",
        "result": {"echo": "build a todo app", "prompt": "review"},
    }


def test_run_returns_400_for_validation_error(client):
    response = client.post("/systems/fake-system/run", json={})

    assert response.status_code == 400
    assert response.get_json() == {"error": "'user_prompt' is required"}


def test_run_returns_404_for_unknown_system(client):
    response = client.post("/systems/unknown/run", json={"user_prompt": "hello"})

    assert response.status_code == 404
    assert response.get_json() == {"error": "Unknown system_id 'unknown'"}


def test_run_returns_500_for_unexpected_error(client):
    response = client.post("/systems/fake-system/run", json={"user_prompt": "explode"})

    assert response.status_code == 500
    assert response.get_json() == {"error": "Execution failed", "details": "boom"}


def test_tool_callback_acknowledges_payload(client):
    response = client.post("/agent/agent-1/tool/python", json={"output": "done"})

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "received",
        "agent_id": "agent-1",
        "tool_name": "python",
    }


def test_tool_output_callback_acknowledges_payload(client):
    response = client.post("/agent/agent-1/tool/python/output", json={"output": "done"})

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "received",
        "agent_id": "agent-1",
        "tool_name": "python",
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

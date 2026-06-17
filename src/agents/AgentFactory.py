from typing import Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr
from src.settings import settings

from src.agents.constants import AgentType

class AgentFactory:
    @staticmethod
    def build_agent(
        prompt: str,
        tools: list,
        temperature: float = 0.7,
        model_name: str = "gpt-5.4-mini",
        middleware: Sequence[AgentMiddleware] | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        response_format=None,
    ):

        assert settings.OPEN_ROUTER_KEY is not None, "OPEN_ROUTER_KEY must be set in the .env file"

        llm = ChatOpenAI(
            api_key=SecretStr(settings.OPEN_ROUTER_KEY),
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            temperature=temperature,
        )

        agent = create_agent(
            model=llm,
            tools=tools or [],
            system_prompt=prompt,
            middleware=middleware or (),
            checkpointer=checkpointer,
            response_format=response_format,
        )
        
        return agent
    @staticmethod
    def build_agent_with_HITLmiddleware_InMemCheckpointer(
        prompt: str,
        tools: list,
        temperature: float = 0.7,
        model_name: str = "gpt-5.4-mini",
        response_format=None,
    ):
        middleware = [
            HumanInTheLoopMiddleware(
                interrupt_on={
                    tool.name: {"allowed_decisions": ["approve", "reject"]}
                    for tool in tools
                }
            )
        ]
        checkpointer = InMemorySaver()
        return AgentFactory.build_agent(
            prompt=prompt,
            tools=tools,
            temperature=temperature,
            model_name=model_name,
            middleware=middleware,
            checkpointer=checkpointer,
            response_format=response_format,
        )

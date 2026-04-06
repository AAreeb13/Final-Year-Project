
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import SecretStr


def import_settings():
    from src.settings import settings
    return settings


if __name__ == "__main__":
    settings = import_settings()

    assert settings.OPEN_ROUTER_KEY is not None, "openrouterkey must be set in the .env file"
    print("Settings imported successfully. openrouterkey is set.")
    model = ChatOpenAI(
        api_key=SecretStr(settings.OPEN_ROUTER_KEY),
        base_url="https://openrouter.ai/api/v1",
        model="gpt-4o-mini",
        temperature=0.7,
        max_completion_tokens=1000)
    agent = create_agent(
        model=model,
        system_prompt="You are a precise software designing agent, that designs software based on SOLID design principles." + 
        "Highlight key components and relationships between components"
        "You are explicited expected to refrain from writing anything other than Components, relationships, and SOLID principles that are used."
    )
    # print("here")
    response = agent.invoke({"messages": [HumanMessage("Chess")]})
    print(response["messages"][-1].content)
 
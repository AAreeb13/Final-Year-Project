from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict 

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    OPEN_ROUTER_KEY: str | None = None
    GITHUB_PERSONAL_ACCESS_TOKEN: str | None = None
    WORKPLACE_FOLDER: str | None = None
    AGENT_SYSTEM_CONTAINER_NAME: str = "agent_system_container"
    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: str | None = None
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "final-year-project-single-agent"
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.OPEN_ROUTER_KEY is None:
            raise ValueError("OPEN_ROUTER_KEY must be set in the .env file")
        if self.GITHUB_PERSONAL_ACCESS_TOKEN is None:
            print("Warning: GITHUB_PERSONAL_ACCESS_TOKEN is not set. The agent will not be able to use GitHub-related tools.")
        if self.WORKPLACE_FOLDER is None:
            raise ValueError("WORKPLACE_FOLDER must be set in the .env file")

settings = Settings()

if __name__ == "__main__":
    assert settings.OPEN_ROUTER_KEY is not None, "OPEN_ROUTER_KEY must be set in the .env file"
    assert settings.AGENT_SYSTEM_CONTAINER_NAME is not None, "AGENT_SYSTEM_CONTAINER_NAME must be set in the .env file"

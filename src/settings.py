from pydantic_settings import BaseSettings, SettingsConfigDict 

class Settings(BaseSettings):
    OPEN_ROUTER_KEY: str
    GITHUB_PERSONAL_ACCESS_TOKEN: str
    WORKPLACE_FOLDER: str
    AGENT_SYSTEM_CONTAINER_NAME: str
    model_config = SettingsConfigDict(
        env_file=".env",
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
            self.WORKPLACE_FOLDER = input("Enter the path to the workplace folder: ").strip()
            if not self.WORKPLACE_FOLDER:
                raise ValueError("WORKPLACE_FOLDER must be set in the .env file")
        if self.AGENT_SYSTEM_CONTAINER_NAME is None:
            self.AGENT_SYSTEM_CONTAINER_NAME = "agent_system_container"

settings = Settings()

if __name__ == "__main__":
    assert settings.OPEN_ROUTER_KEY is not None, "OPEN_ROUTER_KEY must be set in the .env file"
    assert settings.AGENT_SYSTEM_CONTAINER_NAME is not None, "AGENT_SYSTEM_CONTAINER_NAME must be set in the .env file"
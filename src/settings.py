from pydantic_settings import BaseSettings, SettingsConfigDict 

class Settings(BaseSettings):
    OPEN_ROUTER_KEY: str
    GITHUB_PERSONAL_ACCESS_TOKEN: str
    WORKPLACE_FOLDER: str
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
            raise ValueError("WORKPLACE_FOLDER must be set in the .env file")

settings = Settings()

if __name__ == "__main__":
    assert settings.OPEN_ROUTER_KEY is not None, "OPEN_ROUTER_KEY must be set in the .env file"

from pydantic_settings import BaseSettings, SettingsConfigDict 

class Settings(BaseSettings):
    OPEN_ROUTER_KEY: str
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.OPEN_ROUTER_KEY is None:
            raise ValueError("OPEN_ROUTER_KEY must be set in the .env file")
        
    
    
settings = Settings()

if __name__ == "__main__":
    assert settings.OPEN_ROUTER_KEY is not None, "OPEN_ROUTER_KEY must be set in the .env file"

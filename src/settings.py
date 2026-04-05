from pydantic_settings import BaseSettings, SettingsConfigDict 

class Settings(BaseSettings):
    openrouterkey: str
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.openrouterkey is None:
            raise ValueError("openrouterkey must be set in the .env file")
        
    
    
settings = Settings()

if __name__ == "__main__":
    assert settings.openrouterkey is not None, "openrouterkey must be set in the .env file"

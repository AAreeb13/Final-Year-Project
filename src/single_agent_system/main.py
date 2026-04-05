

def import_settings():
    from src.settings import settings
    return settings


if __name__ == "__main__":
    settings = import_settings()
    assert settings.openrouterkey is not None, "openrouterkey must be set in the .env file"
    print("Settings imported successfully. openrouterkey is set.")
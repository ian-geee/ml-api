from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    allowed_origins: List[str] = [
        "https://ian-geee.github.io",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ]
    model_dir: str = "app/models"
    ml_api_key: str | None = None
    class Config: env_file = ".env"

settings = Settings()

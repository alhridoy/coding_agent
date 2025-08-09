from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    anthropic_api_key: str
    github_token: str
    e2b_api_key: str
    daytona_api_key: Optional[str] = None

    # GitHub App Authentication (optional, for enhanced security)
    github_app_id: Optional[str] = None
    github_app_installation_id: Optional[str] = None
    github_app_private_key: Optional[str] = None
    
    # Observability
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "project-tracing"
    langsmith_tracing: bool = True
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    
    # Application Settings
    max_concurrent_sessions: int = 5
    sandbox_timeout_minutes: int = 30
    
    # Development
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # Allow extra fields in .env


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

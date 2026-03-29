"""
Configuration loader for Market Research Aggregator.
Reads .env file and provides typed settings via Pydantic.
"""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).parent / ".env")


class Settings(BaseSettings):
    """All application settings with defaults and validation."""

    # --- API Keys ---
    groq_api_key: str = Field(default="", description="Groq API key")
    serper_api_key: str = Field(default="", description="Serper.dev API key")

    # --- Model Config ---
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # --- Search Config ---
    max_search_queries: int = Field(default=8, ge=1, le=20)
    search_results_per_query: int = Field(default=5, ge=1, le=10)
    max_articles_to_scrape: int = Field(default=15, ge=1, le=50)

    # --- RAG Config ---
    chunk_size: int = Field(default=1000, ge=200, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    top_k_chunks: int = Field(default=20, ge=5, le=50)

    # --- Output ---
    output_dir: str = Field(default="./reports")

    # --- MCP Server ---
    mcp_server_port: int = Field(default=3001)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Return validated settings instance."""
    settings = Settings()
    return settings


def validate_api_keys(settings: Settings) -> list[str]:
    """Check that required API keys are set. Returns list of missing keys."""
    missing = []
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        missing.append("GROQ_API_KEY")
    if not settings.serper_api_key or settings.serper_api_key == "your_serper_api_key_here":
        missing.append("SERPER_API_KEY")
    return missing

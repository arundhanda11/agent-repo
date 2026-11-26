"""Load environment configuration for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_MODEL = "qwen/qwen3.8-27b"


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings populated from environment variables."""

    groq_api_key: str
    serpapi_api_key: str | None
    model: str
    max_tool_calls: int = 3
    max_history_messages: int = 20
    search_results: int = 5
    search_timeout_seconds: float = 30.0
    retry_backoff_seconds: float = 1.0

    @property
    def web_search_enabled(self) -> bool:
        """Whether live search can be used."""
        return bool(self.serpapi_api_key)


def load_settings(
    *, require_groq: bool = True, env_file: str | os.PathLike[str] | None = None
) -> Settings:
    """Read .env (real environment variables take precedence) and return settings."""
    load_dotenv(dotenv_path=env_file, override=False)

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    serpapi_api_key = os.getenv("SERPAPI_API_KEY", "").strip() or None

    if require_groq and not groq_api_key:
        raise ConfigurationError(
            "GROQ_API_KEY is missing. Add it to .env (see .env.example) or set it "
            "in your environment."
        )

    return Settings(
        groq_api_key=groq_api_key,
        serpapi_api_key=serpapi_api_key,
        model=os.getenv("GROQ_MODEL", "").strip() or DEFAULT_MODEL,
    )

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


def _get_setting(name: str) -> str:
    """Read a setting from the environment, falling back to st.secrets."""
    value = os.getenv(name, "").strip()
    if value:
        return value

    try:
        import streamlit as st

        secret = st.secrets.get(name, "")
    except Exception:
        return ""

    return str(secret).strip() if secret is not None else ""


def load_settings(
    *, require_groq: bool = True, env_file: str | os.PathLike[str] | None = None
) -> Settings:
    """Read settings from environment variables (including .env) or st.secrets."""
    load_dotenv(dotenv_path=env_file, override=False)

    groq_api_key = _get_setting("GROQ_API_KEY")
    serpapi_api_key = _get_setting("SERPAPI_API_KEY") or None

    if require_groq and not groq_api_key:
        raise ConfigurationError(
            "GROQ_API_KEY is missing. Set it in your environment, add it to .env "
            "(see .env.example), or add it to st.secrets."
        )

    return Settings(
        groq_api_key=groq_api_key,
        serpapi_api_key=serpapi_api_key,
        model=_get_setting("GROQ_MODEL") or DEFAULT_MODEL,
    )

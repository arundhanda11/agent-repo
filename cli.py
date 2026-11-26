"""Simple terminal chat client for the agent."""

from __future__ import annotations

import sys

from agent import Agent
from config import ConfigurationError, load_settings


def main() -> int:
    """Run the interactive terminal loop."""
    try:
        settings = load_settings()
        agent = Agent(settings)
    except ConfigurationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("AI Agent (type 'exit' to quit, 'new' to start a new chat)")
    print(f"Model: {settings.model}")
    if not settings.web_search_enabled:
        print("Note: SERPAPI_API_KEY is not set; live web search is disabled.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() in {"new", "new chat"}:
            agent.new_chat()
            print("Started a new chat.")
            continue
        if not user_input:
            continue

        print("Assistant: ", end="", flush=True)
        for event in agent.stream(user_input):
            if event.kind == "token":
                print(event.text, end="", flush=True)
            elif event.kind == "status":
                print(f"\n[{event.text}] ", end="", flush=True)
            elif event.kind == "error":
                print(f"\nError: {event.text}", file=sys.stderr)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

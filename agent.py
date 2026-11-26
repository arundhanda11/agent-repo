"""Streaming agent loop: Groq tool calling, bounded search, and memory."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    GroqError,
    InternalServerError,
    RateLimitError,
)
from groq import Groq

from config import ConfigurationError, Settings, load_settings
from tools import (
    SearchResult,
    WEB_SEARCH_TOOLS,
    SearchError,
    format_search_results,
    format_sources,
    web_search,
)

RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)


@dataclass(frozen=True)
class AgentEvent:
    """A streamed update for the UI or CLI."""

    kind: Literal["status", "token", "error"]
    text: str = ""


@dataclass(frozen=True)
class ToolCall:
    """A tool request assembled from streamed deltas."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class CompletionResult:
    """The text and tool calls from one streamed completion."""

    text: str
    tool_calls: list[ToolCall]


class Agent:
    """A conversational assistant that searches only when the model asks."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Groq | None = None,
        search_fn: Callable[[str], list[SearchResult]] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        if not self.settings.groq_api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is missing. Add it to .env (see .env.example) or set "
                "it in your environment."
            )
        self.client = client or Groq(api_key=self.settings.groq_api_key)
        self._search_fn = search_fn or (
            lambda query: web_search(query, self.settings)
        )
        self._sleep = sleep or time.sleep
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()}
        ]

    def new_chat(self) -> None:
        """Clear conversation history while keeping the system prompt."""
        self.messages = [{"role": "system", "content": self._system_prompt()}]

    def stream(self, user_message: str) -> Iterator[AgentEvent]:
        """Answer one question, yielding status, token, and error events."""
        message = user_message.strip()
        if not message:
            yield AgentEvent("error", "Please enter a message.")
            return

        snapshot = len(self.messages)
        self._append({"role": "user", "content": message})
        try:
            answer, _ = yield from self._run()
        except ConfigurationError as exc:
            del self.messages[snapshot:]
            yield AgentEvent("error", str(exc))
            return
        except SearchError as exc:
            del self.messages[snapshot:]
            yield AgentEvent("error", str(exc))
            return
        except RateLimitError:
            del self.messages[snapshot:]
            yield AgentEvent(
                "error", "Groq rate limit reached. Please wait a moment and try again."
            )
            return
        except RETRYABLE_ERRORS as exc:
            del self.messages[snapshot:]
            yield AgentEvent("error", f"Groq connection failed: {exc}")
            return
        except APIStatusError as exc:
            del self.messages[snapshot:]
            detail = getattr(exc, "message", None) or str(exc)
            yield AgentEvent("error", f"Groq API error ({exc.status_code}): {detail}")
            return
        except GroqError as exc:
            del self.messages[snapshot:]
            yield AgentEvent("error", f"Groq API error: {exc}")
            return
        except Exception as exc:
            del self.messages[snapshot:]
            yield AgentEvent("error", f"Unexpected error: {exc}")
            return

        self._append({"role": "assistant", "content": answer})
        self._trim_history()

    def _run(self) -> Generator[AgentEvent, None, tuple[str, list[SearchResult]]]:
        used = 0
        sources: list[SearchResult] = []

        while True:
            can_search = used < self.settings.max_tool_calls
            completion = yield from self._complete(
                self.messages, tools=WEB_SEARCH_TOOLS
            )
            calls = completion.tool_calls

            if not calls:
                answer = (
                    completion.text.strip()
                    or "I couldn't generate a response. Please try again."
                )
                source_block = self._source_block(answer, sources)
                if source_block:
                    yield AgentEvent("token", source_block)
                    answer += source_block
                return answer, sources

            if not can_search:
                # Some models keep requesting tools after the cap; stop and answer.
                answer = completion.text.strip()
                if not answer:
                    answer = self._fallback_answer(sources)
                    yield AgentEvent("token", answer)
                source_block = self._source_block(answer, sources)
                if source_block:
                    yield AgentEvent("token", source_block)
                    answer += source_block
                return answer, sources

            self._append(
                {
                    "role": "assistant",
                    "content": completion.text or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                        for call in calls
                    ],
                }
            )

            for call in calls:
                if used >= self.settings.max_tool_calls:
                    content = (
                        "Search limit reached (3 searches per question). Answer "
                        "using the information already available."
                    )
                elif call.name != "web_search":
                    content = f"Unknown tool '{call.name}'. Only web_search is available."
                else:
                    query = self._parse_query(call.arguments)
                    if query is None:
                        content = (
                            "Invalid tool arguments: expected a JSON object with a "
                            "'query' string."
                        )
                    else:
                        used += 1
                        yield AgentEvent("status", "Searching the web...")
                        try:
                            found = self._search_fn(query)
                        except SearchError as exc:
                            content = f"Search failed: {exc}"
                        else:
                            if found:
                                sources.extend(found)
                                content = format_search_results(found)
                            else:
                                content = (
                                    "No results found for this query. Be honest if "
                                    "you cannot verify the answer."
                                )
                self._append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": content,
                    }
                )

    def _complete(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None
    ) -> Generator[AgentEvent, None, CompletionResult]:
        for attempt in range(2):
            text = ""
            calls: dict[int, dict[str, str]] = {}
            try:
                kwargs: dict[str, Any] = {
                    "model": self.settings.model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_completion_tokens": 2048,
                    "stream": True,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                stream = self.client.chat.completions.create(**kwargs)
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        text += delta.content
                        yield AgentEvent("token", delta.content)
                    for delta_call in delta.tool_calls or []:
                        index = delta_call.index or 0
                        entry = calls.setdefault(
                            index, {"id": "", "name": "", "arguments": ""}
                        )
                        if delta_call.id:
                            entry["id"] = delta_call.id
                        if delta_call.function:
                            if delta_call.function.name:
                                entry["name"] += delta_call.function.name
                            if delta_call.function.arguments:
                                entry["arguments"] += delta_call.function.arguments

                tool_calls = [
                    ToolCall(
                        id=data["id"] or f"call_{index}",
                        name=data["name"],
                        arguments=data["arguments"],
                    )
                    for index, data in sorted(calls.items())
                ]
                return CompletionResult(text=text, tool_calls=tool_calls)
            except RETRYABLE_ERRORS:
                # Retry only before emitting tokens, so the UI never duplicates output.
                if attempt == 0 and not text:
                    yield AgentEvent("status", "Rate limit hit; retrying...")
                    self._sleep(self.settings.retry_backoff_seconds)
                    continue
                raise

        raise RuntimeError("Unreachable retry state.")

    @staticmethod
    def _parse_query(arguments: str) -> str | None:
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(parsed, str) and parsed.strip():
            return parsed.strip()
        if not isinstance(parsed, dict):
            return None
        for key in ("query", "q", "search_query"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        values = [
            value.strip()
            for value in parsed.values()
            if isinstance(value, str) and value.strip()
        ]
        return values[0] if len(values) == 1 else None

    @staticmethod
    def _fallback_answer(sources: list[SearchResult]) -> str:
        if not sources:
            return "I could not verify this with web search. Please try a more specific question."
        lines = ["Here is what I found:"]
        seen: set[str] = set()
        for result in sources:
            if result.link in seen:
                continue
            seen.add(result.link)
            summary = f"- [{result.title}]({result.link})"
            if result.snippet:
                summary += f": {' '.join(result.snippet.split())[:200]}"
            lines.append(summary)
            if len(seen) >= 5:
                break
        return "\n".join(lines)

    @staticmethod
    def _source_block(answer: str, sources: list[SearchResult]) -> str:
        if not sources or "sources:" in answer.lower():
            return ""
        missing: list[SearchResult] = []
        seen: set[str] = set()
        for result in sources:
            if result.link in seen:
                continue
            seen.add(result.link)
            if result.link not in answer:
                missing.append(result)
        return format_sources(missing) if missing else ""

    def _system_prompt(self) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return (
            "You are a fast, accurate, helpful AI assistant.\n"
            f"Today's date (UTC) is {today}.\n"
            "- Answer directly for stable topics: general knowledge, coding, math, "
            "and advice. Do not search for these.\n"
            "- Use web_search only for time-sensitive or unverified information: "
            "news, current events, prices, weather, sports, or recent facts.\n"
            "- Be concise, accurate, and honest. If unsure or the results do not "
            "answer the question, say so instead of guessing.\n"
            "- Reply in the same language as the user (English, Hindi, or Hinglish).\n"
            "- When you search, base the answer on the results and end with a short "
            "Sources section containing links from those results.\n"
            "- Call web_search with exactly {\"query\": \"your focused query\"}.\n"
            "- Prefer one focused search; search again only if the first result is "
            "clearly incomplete.\n"
            f"- You may call web_search at most {self.settings.max_tool_calls} times "
            "per question."
        )

    def _append(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def _trim_history(self) -> None:
        """Keep the system prompt plus the last N messages on a turn boundary."""
        history = self.messages[1:]
        if len(history) <= self.settings.max_history_messages:
            return
        start = len(history) - self.settings.max_history_messages
        while start > 0 and history[start]["role"] != "user":
            start -= 1
        self.messages = [self.messages[0], *history[start:]]

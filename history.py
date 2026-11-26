"""Persistent multi-chat history stored locally as JSON."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HISTORY_FILE = Path(".streamlit/chat_history.json")
UPLOAD_DIR = Path(".streamlit/uploads")

Message = dict[str, Any]
Chat = dict[str, Any]
Store = dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_store() -> Store:
    return {"active_id": None, "chats": []}


def load_store(path: Path = HISTORY_FILE) -> Store:
    """Load history, falling back to an empty store if the file is invalid."""
    try:
        store = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_store()
    if not isinstance(store, dict) or not isinstance(store.get("chats"), list):
        return empty_store()
    return store


def save_store(store: Store, path: Path = HISTORY_FILE) -> None:
    """Atomically write history so a restart never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def new_chat(store: Store, path: Path = HISTORY_FILE) -> Chat:
    chat: Chat = {
        "id": uuid.uuid4().hex,
        "title": "New chat",
        "created_at": _now(),
        "updated_at": _now(),
        "messages": [],
    }
    store["chats"].insert(0, chat)
    store["active_id"] = chat["id"]
    save_store(store, path)
    return chat


def set_active(store: Store, chat_id: str, path: Path = HISTORY_FILE) -> None:
    store["active_id"] = chat_id
    save_store(store, path)


def get_chat(store: Store, chat_id: str | None) -> Chat | None:
    return next((chat for chat in store["chats"] if chat["id"] == chat_id), None)


def messages_for(store: Store, chat_id: str | None) -> list[Message]:
    chat = get_chat(store, chat_id)
    return list(chat["messages"]) if chat else []


def append_message(
    store: Store,
    chat_id: str | None,
    role: str,
    content: str,
    image: str | None = None,
    path: Path = HISTORY_FILE,
) -> Message | None:
    """Append a message and name the chat from its first user message."""
    chat = get_chat(store, chat_id)
    if not chat:
        return None

    first_user = role == "user" and not any(
        message["role"] == "user" for message in chat["messages"]
    )
    message: Message = {"role": role, "content": content}
    if image:
        message["image"] = image
    chat["messages"].append(message)

    if first_user:
        title = " ".join(content.split())[:40]
        chat["title"] = title or "Photo chat"
    chat["updated_at"] = _now()
    save_store(store, path)
    return message


def delete_message(
    store: Store, chat_id: str | None, index: int, path: Path = HISTORY_FILE
) -> Message | None:
    """Remove one message from a chat and return it."""
    chat = get_chat(store, chat_id)
    if not chat or not 0 <= index < len(chat["messages"]):
        return None
    message = chat["messages"].pop(index)
    chat["updated_at"] = _now()
    save_store(store, path)
    return message


def delete_chat(store: Store, chat_id: str, path: Path = HISTORY_FILE) -> None:
    store["chats"] = [chat for chat in store["chats"] if chat["id"] != chat_id]
    if store.get("active_id") == chat_id:
        store["active_id"] = store["chats"][0]["id"] if store["chats"] else None
    save_store(store, path)


def save_upload(upload: Any, suffix: str = "jpg") -> str:
    """Persist a camera or gallery image and return its relative path."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}.{suffix.lstrip('.')}"
    if hasattr(upload, "getvalue"):
        path.write_bytes(upload.getvalue())
    else:
        upload.save(path)
    return str(path)


def sorted_chats(store: Store) -> list[Chat]:
    return sorted(store["chats"], key=lambda chat: chat["updated_at"], reverse=True)

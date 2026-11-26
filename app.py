"""Streamlit chat UI with persistent chats, voice replies, and media options."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.components.v1 import html as html_component

import history
from agent import Agent
from config import ConfigurationError, load_settings

SCREENSHOT_LIBRARY = (
    "https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"
)


def speakable_text(markdown: str) -> str:
    """Strip Markdown/URLs so the browser reads only the spoken words."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown)
    text = re.sub(r"https?://\S+", "", text)
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"^\s*(?:[-*>#]+|\d+\.)\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{2,}", "\n", text).strip()


def voice_language(text: str) -> str:
    """Use a Hindi voice when the reply contains Devanagari script."""
    return "hi-IN" if re.search(r"[\u0900-\u097f]", text) else "en-US"


def speak(text: str) -> None:
    """Ask the browser to speak the final answer using its speech synthesis."""
    clean = speakable_text(text)
    if not clean:
        return
    safe_text = json.dumps(clean).replace("</", "<\\/")
    safe_lang = json.dumps(voice_language(clean))
    script = f"""
    <script>
    (function() {{
      if (!('speechSynthesis' in window)) return;
      const text = {safe_text};
      const lang = {safe_lang};
      window.__agentLastSpeech = {{ text: text, lang: lang }};
      if (window.__agentMuted) return;
      if (typeof window.__agentSpeak === 'function') {{
        window.__agentSpeak(text, lang);
        return;
      }}
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = lang;
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }})();
    </script>
    """
    st.html(script, unsafe_allow_javascript=True)


def render_voice_control() -> None:
    """Show a JS-only mute toggle that cancels speech without a Streamlit rerun."""
    st.html(
        """
        <div style="display: flex; justify-content: flex-end; margin-bottom: 0.5rem;">
          <button id="agent-voice-toggle">Mute voice</button>
        </div>
        <script>
        (function() {
          if (window.__agentMuted === undefined) window.__agentMuted = false;
          if (window.__agentLastSpeech === undefined) window.__agentLastSpeech = null;
          const button = document.getElementById('agent-voice-toggle');
          window.__agentSpeak = function(text, lang) {
            if (!('speechSynthesis' in window)) return;
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = lang || 'en-US';
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
          };
          function paint() {
            button.textContent = window.__agentMuted ? 'Unmute voice' : 'Mute voice';
          }
          button.onclick = function() {
            window.__agentMuted = !window.__agentMuted;
            if (window.__agentMuted) {
              if ('speechSynthesis' in window) window.speechSynthesis.cancel();
            } else if (window.__agentLastSpeech) {
              window.__agentSpeak(
                window.__agentLastSpeech.text,
                window.__agentLastSpeech.lang
              );
            }
            paint();
          };
          paint();
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def render_screenshot() -> None:
    """Render a one-shot browser screenshot script that downloads a PNG."""
    script = f"""
    <script src="{SCREENSHOT_LIBRARY}"></script>
    <script>
    (async function() {{
      if (typeof html2canvas === 'undefined') return;
      const canvas = await html2canvas(
        window.parent.document.body, {{scale: 1, useCORS: true}}
      );
      const link = document.createElement('a');
      link.download = 'agent-chat-screenshot.png';
      link.href = canvas.toDataURL('image/png');
      link.click();
    }})();
    </script>
    """
    html_component(script, height=0)


def transcribe_audio(upload: Any, client: Any) -> str:
    """Transcribe a spoken command with Groq Whisper."""
    media_type = getattr(upload, "type", "") or "audio/webm"
    suffix = media_type.split("/")[-1].split(";")[0] or "webm"
    with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as handle:
        handle.write(upload.getvalue())
        path = Path(handle.name)
    try:
        with path.open("rb") as audio:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo", file=audio
            )
        return (result.text or "").strip()
    finally:
        path.unlink(missing_ok=True)


def load_agent_memory(agent: Agent, messages: list[history.Message]) -> None:
    """Rebuild the in-memory conversation for a selected saved chat."""
    agent.new_chat()
    for message in messages:
        agent.messages.append({"role": message["role"], "content": message["content"]})
    agent._trim_history()


st.set_page_config(page_title="AI Agent", layout="centered")
st.title("AI Agent")

try:
    settings = load_settings()
except ConfigurationError as exc:
    st.error(str(exc))
    st.stop()

if "store" not in st.session_state:
    st.session_state.store = history.load_store()
    store = st.session_state.store
    active = store.get("active_id")
    if not history.get_chat(store, active):
        recent = history.sorted_chats(store)
        active = recent[0]["id"] if recent else history.new_chat(store)["id"]
    st.session_state.current_chat_id = active
    history.set_active(store, active)

store = st.session_state.store
chat_id = st.session_state.current_chat_id

if "agent" not in st.session_state:
    st.session_state.agent = Agent(settings)
if "messages" not in st.session_state:
    st.session_state.messages = history.messages_for(store, chat_id)
    load_agent_memory(st.session_state.agent, st.session_state.messages)
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True
if "last_spoken" not in st.session_state:
    st.session_state.last_spoken = -1
if "screenshot_requested" not in st.session_state:
    st.session_state.screenshot_requested = False
render_voice_control()

st.markdown(
    """
    <style>
      button[title="Remove photo"], button[aria-label="Remove photo"] {
        font-size: 18px;
        line-height: 1;
        padding: 2px 6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.caption(f"Model: {settings.model}")
st.sidebar.header("Chats")

if st.sidebar.button("New chat", use_container_width=True):
    chat = history.new_chat(store)
    st.session_state.current_chat_id = chat["id"]
    st.session_state.messages = []
    st.session_state.agent.new_chat()
    st.session_state.last_spoken = -1
    st.rerun()

for chat in history.sorted_chats(store):
    cols = st.sidebar.columns([4, 1], gap="small")
    if cols[0].button(
        chat["title"][:35], key=f"open_{chat['id']}", use_container_width=True
    ):
        st.session_state.current_chat_id = chat["id"]
        st.session_state.messages = history.messages_for(store, chat["id"])
        load_agent_memory(st.session_state.agent, st.session_state.messages)
        history.set_active(store, chat["id"])
        st.session_state.last_spoken = -1
        st.rerun()
    if cols[1].button("x", key=f"delete_{chat['id']}", use_container_width=True):
        history.delete_chat(store, chat["id"])
        if st.session_state.current_chat_id == chat["id"]:
            remaining = history.sorted_chats(store)
            if remaining:
                st.session_state.current_chat_id = remaining[0]["id"]
                st.session_state.messages = history.messages_for(
                    store, remaining[0]["id"]
                )
                load_agent_memory(st.session_state.agent, st.session_state.messages)
                history.set_active(store, remaining[0]["id"])
            else:
                fresh = history.new_chat(store)
                st.session_state.current_chat_id = fresh["id"]
                st.session_state.messages = []
                st.session_state.agent.new_chat()
        st.session_state.last_spoken = -1
        st.rerun()

st.sidebar.toggle("Voice replies", key="voice_enabled")
if st.sidebar.button(
    "Replay voice", disabled=not st.session_state.messages, use_container_width=True
):
    st.session_state.last_spoken = -1
    st.rerun()

if not settings.web_search_enabled:
    st.sidebar.warning("SERPAPI_API_KEY is not set; live web search is disabled.")

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message.get("image") and Path(message["image"]).exists():
            image_col, remove_col = st.columns(
                [6, 1], gap="small", vertical_alignment="top"
            )
            with image_col:
                st.image(message["image"], width=280)
            with remove_col:
                photo_key = "remove_" + hashlib.md5(
                    str(message["image"]).encode()
                ).hexdigest()[:10]
                if st.button(
                    "×", key=photo_key, help="Remove photo", use_container_width=True
                ):
                    removed = history.delete_message(store, chat_id, index)
                    if removed and removed.get("image"):
                        Path(removed["image"]).unlink(missing_ok=True)
                    st.session_state.messages = history.messages_for(store, chat_id)
                    load_agent_memory(st.session_state.agent, st.session_state.messages)
                    st.rerun()
        if message.get("content"):
            st.markdown(message["content"])

def attach_image(caption: str, upload: object, suffix: str) -> None:
    """Save a camera/gallery image into the current chat and refresh."""
    path = history.save_upload(upload, suffix)
    history.append_message(store, chat_id, "user", caption, image=path)
    st.session_state.messages = history.messages_for(store, chat_id)
    st.session_state.agent.messages.append({"role": "user", "content": caption})
    st.session_state.agent._trim_history()
    st.rerun()


left_col, input_col = st.columns(
    [0.12, 0.88], gap="small", vertical_alignment="bottom"
)

with left_col:
    with st.popover("+", use_container_width=True):
        camera_tab, photos_tab, screenshot_tab = st.tabs(
            ["Camera", "Photos", "Screenshot"]
        )
        with camera_tab:
            camera = st.camera_input("Take a live photo", key="camera_input")
            if camera and st.button("Attach camera photo", key="attach_camera"):
                attach_image("Photo from camera", camera, "jpg")
        with photos_tab:
            upload = st.file_uploader(
                "Choose from gallery",
                type=["png", "jpg", "jpeg", "webp"],
                key="photo_uploader",
            )
            if upload:
                suffix = (getattr(upload, "type", "") or "image/jpeg").split("/")[-1]
                if st.button("Attach selected photo", key="attach_photo"):
                    attach_image("Uploaded photo", upload, suffix)
        with screenshot_tab:
            st.caption("Downloads a PNG of the current page.")
            if st.button("Take a screenshot", key="screenshot_button"):
                st.session_state.screenshot_requested = True
                st.rerun()

with input_col:
    chat_value = st.chat_input(
        "Ask me anything...", accept_audio=True, key="chat_input"
    )

transcribe_box = st.empty()

if chat_value:
    prompt = ""
    if isinstance(chat_value, str):
        prompt = chat_value.strip()
    else:
        prompt = (getattr(chat_value, "text", "") or "").strip()
        audio_files = list(getattr(chat_value, "files", None) or [])
        if audio_files and not prompt:
            transcribe_box.info("Transcribing voice command...")
            try:
                prompt = transcribe_audio(
                    audio_files[0], st.session_state.agent.client
                )
            except Exception as exc:
                transcribe_box.error(f"Could not transcribe audio: {exc}")
            else:
                transcribe_box.empty()
                if not prompt:
                    transcribe_box.warning("No speech detected. Please try again.")

    if prompt:
        history.append_message(store, chat_id, "user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            status_box = st.empty()
            answer = ""

            for event in st.session_state.agent.stream(prompt):
                if event.kind == "status":
                    status_box.info(event.text)
                elif event.kind == "token":
                    status_box.empty()
                    answer += event.text
                    placeholder.markdown(answer + "▌")
                elif event.kind == "error":
                    status_box.empty()
                    answer = event.text
                    placeholder.error(answer)

            if answer:
                placeholder.markdown(answer)
                history.append_message(store, chat_id, "assistant", answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                if st.session_state.voice_enabled:
                    speak(answer)
                    st.session_state.last_spoken = len(st.session_state.messages) - 1

if st.session_state.screenshot_requested:
    st.session_state.screenshot_requested = False
    render_screenshot()

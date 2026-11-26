# AI Agent

A fast, ChatGPT-like assistant powered by the Groq API and SerpAPI. It answers
directly for stable topics and calls `web_search` only when live information is
needed, with streaming replies, memory, and linked sources.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your keys:

```dotenv
GROQ_API_KEY=your_groq_api_key
SERPAPI_API_KEY=your_serpapi_api_key
GROQ_MODEL=qwen/qwen3.8-27b
```

`GROQ_MODEL` is optional. The default is `qwen/qwen3.8-27b`, a current Groq
model with reliable tool calling. `openai/gpt-oss-120b` is a production
alternative. `llama-3.3-70b-versatile` was deprecated for free/developer
accounts on 2026-08-16, so it is no longer the default; set
`GROQ_MODEL=llama-3.3-70b-versatile` only if your account still has access.

## Run

```bash
streamlit run app.py    # browser chat UI
python cli.py           # terminal chat
```

Type `new` in the CLI (or press **New chat** in Streamlit) to clear memory.

## Behavior

- Streaming Groq responses with function calling via the official `groq` SDK.
- `web_search(query)` returns the top 5 SerpAPI organic results; at most 3
  searches per question.
- Only time-sensitive questions trigger search; stable questions are answered
  from model knowledge.
- Sources are appended with links whenever search is used.
- The last 20 messages are kept for follow-ups.
- API failures and rate limits are retried once with backoff; missing keys and
  empty results produce clear messages.
- Replies follow the user's language (English, Hindi, or Hinglish).
- The Streamlit UI speaks each final answer using browser speech synthesis
  (English/Hindi); toggle **Voice replies** or use **Replay voice** in the
  sidebar. The **Mute voice** button at the top stops speech immediately without
  interrupting the answer; click **Unmute voice** to speak the last reply again.
  The browser needs a voice installed for the reply's language.
- Chats are saved locally in `.streamlit/chat_history.json` and survive restarts.
  Each **New chat** gets its own history; switch or delete chats from the sidebar.
- The `+` button sits on the left of the message box and has **Camera** (live
  photo), **Photos** (gallery upload), and **Screenshot** (downloads a PNG of the
  page). Images are stored in `.streamlit/uploads/` and each attached photo has
  an `×` button to remove it from that chat. Camera access needs `localhost` or
  HTTPS.
- Theme is monochrome (black on white), configured in `.streamlit/config.toml`.
- The mic button on the right of the message box records a voice command and
  transcribes it with Groq Whisper (`whisper-large-v3-turbo`); the transcript is
  sent as your message.

If `SERPAPI_API_KEY` is absent, normal answers still work; web search reports
that it is unavailable. `GROQ_API_KEY` is required.

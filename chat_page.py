"""AI Chat page — Carrot Bot, powered by Gemini 2.5 Pro via OpenRouter.

Multi-thread chat (like Claude): sidebar lists chats, "+ New chat" creates one,
clicking switches active. Tool calls go through bot_tools.dispatch and are
silently executed between LLM turns; the user sees only the final natural-
language response.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Iterator, List, Optional

import streamlit as st

from bot_prompt import SYSTEM_PROMPT
from bot_tools import TOOL_SCHEMAS, dispatch

# Try to load .env (local dev) — optional, fine if not present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

MODEL_ID = "google/gemini-2.5-pro"
MAX_TOOL_ROUNDS = 6  # safety: cap recursion of tool calls per user turn

# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------


def _get_api_key() -> Optional[str]:
    """Resolution order: st.secrets → env var. Returns None if not found."""
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return str(st.secrets["OPENROUTER_API_KEY"])
    except Exception:
        pass
    return os.environ.get("OPENROUTER_API_KEY")


@st.cache_resource(show_spinner=False)
def _client():
    """Cached OpenAI client pointed at OpenRouter. Returns None if no key."""
    key = _get_api_key()
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            # OpenRouter likes (but doesn't require) attribution headers
            "HTTP-Referer": "https://github.com/prsuman1/carrot-ladder",
            "X-Title": "Carrot Bot",
        },
    )


# ---------------------------------------------------------------------------
# Chat state model
# ---------------------------------------------------------------------------


def _new_chat_id() -> str:
    return uuid.uuid4().hex[:12]


def _ensure_state() -> None:
    if "chats" not in st.session_state:
        st.session_state["chats"] = {}
    if "active_chat" not in st.session_state:
        cid = _new_chat_id()
        st.session_state["chats"][cid] = _blank_chat()
        st.session_state["active_chat"] = cid


def _blank_chat() -> dict:
    return {
        "title": "New chat",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def _active() -> dict:
    return st.session_state["chats"][st.session_state["active_chat"]]


def _set_title_from_first_user_msg() -> None:
    chat = _active()
    if chat["title"] != "New chat":
        return
    for m in chat["messages"]:
        if m.get("role") == "user":
            text = m["content"][:50].strip().replace("\n", " ")
            chat["title"] = (text + ("…" if len(m["content"]) > 50 else "")) or "New chat"
            return


# ---------------------------------------------------------------------------
# OpenRouter call loop with tool execution
# ---------------------------------------------------------------------------


def stream_assistant_turn() -> Iterator[str]:
    """Stream the assistant's response for the active chat. Internally handles
    any tool calls by executing them and looping back to the model. Yields
    text chunks suitable for st.write_stream.
    """
    client = _client()
    if client is None:
        yield "_OpenRouter API key not configured. See instructions on the right._"
        return

    chat = _active()
    messages = chat["messages"]

    for round_idx in range(MAX_TOOL_ROUNDS):
        try:
            stream = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                tools=TOOL_SCHEMAS,
                stream=True,
                temperature=0.2,
            )
        except Exception as e:
            yield f"\n\n_OpenRouter request failed: `{type(e).__name__}: {e}`_"
            return

        # Aggregate streamed deltas. OpenAI streaming for tool calls comes back
        # as incremental fragments — both `content` and `tool_calls`.
        agg_content: str = ""
        agg_tool_calls: dict = {}  # index → {id, name, args_str}
        finish_reason: Optional[str] = None

        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    agg_content += delta.content
                    yield delta.content  # surface partial text immediately
                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        slot = agg_tool_calls.setdefault(idx, {"id": "", "name": "", "args_str": ""})
                        if tc.id:
                            slot["id"] += tc.id
                        if tc.function and tc.function.name:
                            slot["name"] += tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["args_str"] += tc.function.arguments
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
        except Exception as e:
            yield f"\n\n_Stream interrupted: `{type(e).__name__}: {e}`_"
            return

        # No tool calls → final text turn done.
        if not agg_tool_calls:
            if agg_content:
                messages.append({"role": "assistant", "content": agg_content})
            return

        # Append the assistant's tool-call message (per OpenAI spec).
        oa_tool_calls = [
            {
                "id": v["id"],
                "type": "function",
                "function": {"name": v["name"], "arguments": v["args_str"]},
            }
            for _, v in sorted(agg_tool_calls.items())
        ]
        messages.append({
            "role": "assistant",
            "content": agg_content or None,
            "tool_calls": oa_tool_calls,
        })

        # Execute each tool call and append a tool message.
        for idx, slot in sorted(agg_tool_calls.items()):
            name = slot["name"]
            try:
                args = json.loads(slot["args_str"]) if slot["args_str"].strip() else {}
            except json.JSONDecodeError as e:
                args = {}
                tool_result = {"error": f"could not parse tool arguments: {e}"}
            else:
                tool_result = dispatch(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": slot["id"],
                "content": json.dumps(tool_result, default=str),
            })
        # loop: model gets tool results, may call more tools or produce final text

    yield f"\n\n_(reached {MAX_TOOL_ROUNDS} tool-call rounds without a final answer)_"


# ---------------------------------------------------------------------------
# Sidebar (chat list)
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Carrot Bot")
        st.caption("Ask the data anything. Powered by Gemini 2.5 Pro.")

        if st.button("+ New chat", width="stretch", type="primary"):
            cid = _new_chat_id()
            st.session_state["chats"][cid] = _blank_chat()
            st.session_state["active_chat"] = cid
            st.rerun()

        st.divider()

        chats = st.session_state["chats"]
        active_id = st.session_state["active_chat"]
        # Most recent first
        ordered_ids = sorted(chats.keys(), key=lambda c: chats[c]["created_at"], reverse=True)
        for cid in ordered_ids:
            chat = chats[cid]
            n_msgs = sum(1 for m in chat["messages"] if m.get("role") in ("user", "assistant") and m.get("content"))
            label = chat["title"]
            if cid == active_id:
                label = f"• {label}"
            label = f"{label}  · {n_msgs}"
            if st.button(label, key=f"open_{cid}", width="stretch"):
                st.session_state["active_chat"] = cid
                st.rerun()

        st.divider()
        with st.expander("Manage", expanded=False):
            if st.button("Clear current chat", width="stretch"):
                _active()["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]
                _active()["title"] = "New chat"
                st.rerun()
            if st.button("Delete all chats", width="stretch"):
                st.session_state.pop("chats", None)
                st.session_state.pop("active_chat", None)
                st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------


def _render_no_key_help() -> None:
    st.title("Carrot Bot")
    st.error("OpenRouter API key not found.")
    st.markdown(
        """
**Add your OpenRouter key, then refresh this page.** Three ways:

**1. Streamlit Cloud (production):**
Open the app's "Manage app" → Settings → Secrets, paste:
```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
```

**2. Local `.streamlit/secrets.toml`:**
Create the file at `<repo>/.streamlit/secrets.toml` with the same line.
This file is git-ignored.

**3. Local environment:**
```
export OPENROUTER_API_KEY=sk-or-v1-...
streamlit run app.py
```

Get a key from https://openrouter.ai/settings/keys (Gemini 2.5 Pro is paid; check usage costs in the OpenRouter dashboard).
"""
    )


def _render_chat() -> None:
    chat = _active()
    st.title(chat["title"] if chat["title"] != "New chat" else "Carrot Bot")
    st.caption(f"Model: {MODEL_ID}  ·  {len(TOOL_SCHEMAS)} tools available")

    # Render history (skip system + tool messages from view; show only user/assistant content).
    for m in chat["messages"]:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        if not m.get("content"):
            # Assistant tool-call-only turn — skip rendering, the next assistant text turn covers it.
            continue
        with st.chat_message(role):
            st.markdown(m["content"])

    # Input
    prompt = st.chat_input(
        "Ask Carrot Bot about the ladder, the data, or any what-if…",
    )
    if prompt:
        chat["messages"].append({"role": "user", "content": prompt})
        _set_title_from_first_user_msg()
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.write_stream(stream_assistant_turn())
        st.rerun()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
_ensure_state()
_render_sidebar()
if _get_api_key() is None:
    _render_no_key_help()
else:
    _render_chat()

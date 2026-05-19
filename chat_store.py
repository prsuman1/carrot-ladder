"""Carrot Bot chat persistence via the GitHub REST API.

Reads and writes a single CSV at `data/chats.csv` on the main branch of the
configured repo. Storage is API-only (never the local filesystem) so chats
survive Streamlit Cloud reboots, code pushes, and idle eviction — the Cloud
container's disk is ephemeral.

One row per message. The full OpenAI-format message dict is JSON-serialized
into `message_json` so tool_calls and tool results round-trip cleanly. System
messages are NOT stored — they're re-injected from `bot_prompt.SYSTEM_PROMPT`
at load time.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
from datetime import datetime, timezone
from typing import Optional, Tuple

import httpx
import streamlit as st

from bot_prompt import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Config (overridable via st.secrets)
# ---------------------------------------------------------------------------
DEFAULT_REPO = "prsuman1/carrot-ladder"
DEFAULT_PATH = "data/chats.csv"
DEFAULT_BRANCH = "main"

CSV_HEADER = [
    "chat_id",
    "chat_created_at",
    "chat_title",
    "msg_index",
    "msg_role",
    "content_preview",
    "message_json",
]

_GH_API = "https://api.github.com"


# ---------------------------------------------------------------------------
# Secret resolution
# ---------------------------------------------------------------------------
def _from_secrets(key: str) -> Optional[str]:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key)


def _token() -> Optional[str]:
    return _from_secrets("GITHUB_TOKEN")


def _repo() -> str:
    return _from_secrets("GITHUB_REPO") or DEFAULT_REPO


def _path() -> str:
    return _from_secrets("GITHUB_CHATS_PATH") or DEFAULT_PATH


def _branch() -> str:
    return _from_secrets("GITHUB_BRANCH") or DEFAULT_BRANCH


def is_configured() -> bool:
    """True iff a GitHub token is available and the chat store can be used."""
    return _token() is not None


# ---------------------------------------------------------------------------
# GitHub REST API wrapper
# ---------------------------------------------------------------------------
def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "carrot-bot/1.0",
    }


def _contents_url() -> str:
    return f"{_GH_API}/repos/{_repo()}/contents/{_path()}"


def _get_file() -> tuple[Optional[bytes], Optional[str], Optional[str]]:
    """Fetch the CSV file from GitHub.

    Returns (content_bytes, sha, error). If the file doesn't exist, returns
    (None, None, None) — caller treats this as "empty store, fresh start".
    """
    if not is_configured():
        return None, None, "GITHUB_TOKEN not configured"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                _contents_url(),
                params={"ref": _branch()},
                headers=_headers(),
            )
    except Exception as e:
        return None, None, f"network error: {type(e).__name__}: {e}"

    if resp.status_code == 404:
        return None, None, None  # file doesn't exist yet — fine
    if resp.status_code != 200:
        return None, None, f"GET {resp.status_code}: {resp.text[:200]}"

    payload = resp.json()
    sha = payload.get("sha")
    encoded = payload.get("content", "")
    try:
        raw = base64.b64decode(encoded)
    except Exception as e:
        return None, sha, f"could not decode base64: {e}"
    return raw, sha, None


def _put_file(content_bytes: bytes, sha: Optional[str], message: str) -> tuple[bool, Optional[str], Optional[str]]:
    """PUT (create or update) the CSV file. Returns (ok, new_sha, error)."""
    if not is_configured():
        return False, None, "GITHUB_TOKEN not configured"
    body = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("ascii"),
        "branch": _branch(),
    }
    if sha:
        body["sha"] = sha
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.put(
                _contents_url(),
                headers=_headers(),
                json=body,
            )
    except Exception as e:
        return False, None, f"network error: {type(e).__name__}: {e}"

    if resp.status_code in (200, 201):
        new_sha = resp.json().get("content", {}).get("sha")
        return True, new_sha, None
    return False, None, f"PUT {resp.status_code}: {resp.text[:300]}"


# ---------------------------------------------------------------------------
# CSV (de)serialization
# ---------------------------------------------------------------------------
def _serialize_chats(chats: dict) -> bytes:
    """chats: {chat_id: {title, created_at, messages: [openai-format dicts]}}.
    System messages are filtered out (re-injected at load time).
    """
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(CSV_HEADER)

    # Sort chats by created_at ascending so the CSV is stable / readable
    ordered_chats = sorted(chats.items(), key=lambda kv: kv[1].get("created_at", ""))

    for chat_id, chat in ordered_chats:
        msgs = chat.get("messages", []) or []
        kept = [m for m in msgs if m.get("role") != "system"]
        if not kept:
            # Persist a placeholder row so empty "New chat" tabs survive refresh.
            # Without this, clicking "+ New chat" then refreshing loses the tab.
            writer.writerow([
                chat_id,
                chat.get("created_at", ""),
                chat.get("title", "New chat"),
                0,
                "",          # role placeholder
                "",          # preview placeholder
                "{}",        # message_json placeholder (deserialized as empty)
            ])
            continue
        for idx, m in enumerate(kept):
            content = m.get("content") or ""
            preview = content if isinstance(content, str) else json.dumps(content)
            preview = preview.replace("\n", " ").strip()
            if len(preview) > 200:
                preview = preview[:200] + "…"
            writer.writerow([
                chat_id,
                chat.get("created_at", ""),
                chat.get("title", "New chat"),
                idx,
                m.get("role", ""),
                preview,
                json.dumps(m, ensure_ascii=False, default=str),
            ])
    return buf.getvalue().encode("utf-8")


def _deserialize_chats(content_bytes: bytes) -> dict:
    """Parse CSV → {chat_id: chat_dict}. Re-injects SYSTEM_PROMPT as msg 0."""
    text = content_bytes.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    chats: dict[str, dict] = {}
    for row in reader:
        cid = row.get("chat_id")
        if not cid:
            continue
        if cid not in chats:
            chats[cid] = {
                "title": row.get("chat_title") or "New chat",
                "created_at": row.get("chat_created_at") or _utcnow(),
                # Will prepend system prompt below after collecting all rows
                "_msgs": [],
            }
        try:
            m = json.loads(row.get("message_json") or "{}")
        except json.JSONDecodeError:
            # Fall back to a minimal reconstruction
            m = {
                "role": row.get("msg_role") or "user",
                "content": row.get("content_preview") or "",
            }
        # Placeholder rows for empty chats: role/content blank — skip them so
        # the chat starts cleanly with just the system prompt.
        if not m.get("role") and not m.get("content"):
            continue
        chats[cid]["_msgs"].append((int(row.get("msg_index") or 0), m))

    # Sort by msg_index and prepend system prompt
    final: dict[str, dict] = {}
    for cid, info in chats.items():
        ordered = [m for _, m in sorted(info["_msgs"], key=lambda t: t[0])]
        final[cid] = {
            "title": info["title"],
            "created_at": info["created_at"],
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + ordered,
        }
    return final


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_chats() -> dict:
    """Fetch chats.csv from GitHub and return the chats dict (or {} on failure).

    Caches the file's SHA in `st.session_state['__chat_csv_sha__']` for the
    next save's optimistic-concurrency check.
    """
    content, sha, err = _get_file()
    if err:
        # Silent failure — caller treats as "no chats yet"
        st.session_state["__chat_csv_sha__"] = None
        st.session_state["__chat_store_last_error__"] = err
        return {}
    st.session_state["__chat_csv_sha__"] = sha
    st.session_state["__chat_store_last_error__"] = None
    if not content:
        return {}
    try:
        return _deserialize_chats(content)
    except Exception as e:
        st.session_state["__chat_store_last_error__"] = f"parse error: {type(e).__name__}: {e}"
        return {}


def save_chats(chats: dict) -> Tuple[bool, Optional[str]]:
    """Serialize all chats to CSV and PUT to GitHub.

    Returns (success, error_message). On SHA conflict, retries once with a
    fresh SHA. On other errors, returns False + a short error string suitable
    for `st.toast`.
    """
    if not is_configured():
        return False, "GITHUB_TOKEN not configured"

    content = _serialize_chats(chats)
    sha = st.session_state.get("__chat_csv_sha__")
    msg = f"chat: update chats.csv [{_utcnow()}]"

    ok, new_sha, err = _put_file(content, sha, msg)
    if ok:
        st.session_state["__chat_csv_sha__"] = new_sha
        st.session_state["__chat_store_last_error__"] = None
        return True, None

    # Possible 409 (sha conflict) or 422 (mismatched sha). Refresh and retry once.
    if err and ("409" in err or "422" in err or "sha" in err.lower()):
        fresh_content, fresh_sha, fetch_err = _get_file()
        if fetch_err is None:
            st.session_state["__chat_csv_sha__"] = fresh_sha
            ok2, new_sha2, err2 = _put_file(content, fresh_sha, msg)
            if ok2:
                st.session_state["__chat_csv_sha__"] = new_sha2
                return True, None
            st.session_state["__chat_store_last_error__"] = err2
            return False, err2

    st.session_state["__chat_store_last_error__"] = err
    return False, err

from pathlib import Path
from datetime import datetime
import json
import re

from app.core.config import (
    MEMORY_DIR,
    MEMORY_MAX_RECENT_TURNS,
)


def clean_text_for_json(text) -> str:
    """
    Xóa/thay thế các ký tự Unicode lỗi dạng surrogate
    để có thể lưu JSON bằng UTF-8 an toàn.
    """
    if text is None:
        return ""

    text = str(text)

    return text.encode("utf-8", errors="replace").decode("utf-8")


def clean_jsonable(obj):
    """
    Làm sạch toàn bộ dict/list/string trước khi json.dump.
    """
    if isinstance(obj, dict):
        return {
            clean_text_for_json(k): clean_jsonable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [clean_jsonable(item) for item in obj]

    if isinstance(obj, str):
        return clean_text_for_json(obj)

    return obj


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_session_id(session_id: str) -> str:
    session_id = str(session_id or "default")
    return re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)


def format_messages(messages: list[dict]) -> str:
    if not messages:
        return "Không có lịch sử hội thoại."

    lines = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if not content:
            continue

        if role == "user":
            lines.append(f"Người dùng: {content}")
        elif role == "assistant":
            lines.append(f"Trợ lý: {content}")
        else:
            lines.append(f"{role}: {content}")

    if not lines:
        return "Không có lịch sử hội thoại."

    return "\n".join(lines)


class FileChatMemoryStore:
    """
    Lưu memory theo session_id xuống file JSON.

    Cấu trúc mỗi session:
    {
        "session_id": "...",
        "summary": "...",
        "messages": [
            {"role": "user", "content": "...", "created_at": "..."},
            {"role": "assistant", "content": "...", "created_at": "..."}
        ]
    }
    """

    def __init__(self, memory_dir: Path = MEMORY_DIR):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def session_path(self, session_id: str) -> Path:
        clean_id = safe_session_id(session_id)
        return self.memory_dir / f"{clean_id}.json"

    def load(self, session_id: str) -> dict:
        path = self.session_path(session_id)

        if not path.exists():
            return {
                "session_id": safe_session_id(session_id),
                "summary": "",
                "messages": [],
            }

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        except json.JSONDecodeError:
            print(f"Memory file bị lỗi JSON, tạo lại session mới: {path}")
            return {
                "session_id": safe_session_id(session_id),
                "summary": "",
                "messages": [],
            }

    def save(self, session_id: str, session: dict):
        path = self.session_path(session_id)

        clean_session = clean_jsonable(session)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean_session, f, ensure_ascii=False, indent=2)

    def append_message(self, session_id: str, role: str, content: str):
        session = self.load(session_id)

        session.setdefault("messages", []).append(
            {
                "role": clean_text_for_json(role),
                "content": clean_text_for_json(content),
                "created_at": now_iso(),
            }
        )

        self.save(session_id, session)

    def clear(self, session_id: str):
        path = self.session_path(session_id)

        if path.exists():
            path.unlink()

    def get_recent_messages(
        self,
        session: dict,
        max_turns: int = MEMORY_MAX_RECENT_TURNS,
    ) -> list[dict]:
        messages = session.get("messages", [])
        max_messages = max_turns * 2
        return messages[-max_messages:]

    def get_memory_parts(self, session_id: str) -> tuple[str, str]:
        session = self.load(session_id)

        summary = session.get("summary", "").strip()
        if not summary:
            summary = "Chưa có tóm tắt hội thoại."

        recent_messages = self.get_recent_messages(session)
        recent_text = format_messages(recent_messages)

        return summary, recent_text
from __future__ import annotations

import json
import threading
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List


class ConversationMemory:
    """轻量级会话存储：磁盘保存完整历史，提示词只取最近几轮。"""

    def __init__(self, storage_dir: Path, max_rounds: int = 4):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_rounds = max_rounds
        self._lock = threading.Lock()
        self._sessions: Dict[str, Deque[dict]] = defaultdict(deque)

    def new_session_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def get(self, session_id: str) -> List[dict]:
        with self._lock:
            if not self._sessions[session_id]:
                self._load(session_id)
            return list(self._sessions[session_id])

    def append(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            self._sessions[session_id].append({"question": question, "answer": answer})
            self._save(session_id)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions[session_id].clear()
            path = self._path(session_id)
            if path.exists():
                path.unlink()

    def format_context(self, session_id: str) -> str:
        rounds = self.get(session_id)[-self.max_rounds :]
        if not rounds:
            return "无"
        lines = []
        for index, item in enumerate(rounds, start=1):
            lines.append(f"第 {index} 轮用户：{item['question']}")
            lines.append(f"第 {index} 轮助手：{item['answer'][:500]}")
        return "\n".join(lines)

    def _path(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}.json"

    def _load(self, session_id: str) -> None:
        path = self._path(session_id)
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self._sessions[session_id] = deque(data)

    def _save(self, session_id: str) -> None:
        path = self._path(session_id)
        path.write_text(json.dumps(list(self._sessions[session_id]), ensure_ascii=False, indent=2), encoding="utf-8")

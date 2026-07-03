from __future__ import annotations

import json
import threading
import uuid
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional

from .config import Settings


class ConversationMemory:
    def __init__(self, settings: Settings, max_rounds: int = 4):
        self.settings = settings
        self.storage_dir = settings.conversations_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_rounds = max_rounds
        self._lock = threading.Lock()
        self._sessions: Dict[tuple[int, str], Deque[dict]] = defaultdict(deque)
        self._init_schema()

    def _connect(self):
        import pymysql
        return pymysql.connect(
            host=self.settings.mysql_host,
            port=self.settings.mysql_port,
            user=self.settings.mysql_user,
            password=self.settings.mysql_password,
            database=self.settings.mysql_database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )

    def _execute(self, sql: str, params: Iterable[Any] = (), fetchone: bool = False, fetchall: bool = False):
        query = sql.replace("?", "%s")
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(query, tuple(params))
            if fetchone:
                row = cur.fetchone()
                return dict(row) if row else None
            if fetchall:
                return [dict(row) for row in cur.fetchall()]
            return cur.lastrowid
        finally:
            conn.close()

    def _init_schema(self) -> None:
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                session_id VARCHAR(32) PRIMARY KEY COMMENT 'conversation id',
                user_id INT NOT NULL COMMENT 'owner user id',
                title VARCHAR(255) NOT NULL COMMENT 'first user question summary',
                session_type VARCHAR(16) DEFAULT 'chat' COMMENT 'session type: chat/interview/job',
                created_at VARCHAR(32) NOT NULL COMMENT 'created time',
                updated_at VARCHAR(32) NOT NULL COMMENT 'last updated time',
                INDEX idx_conversation_sessions_user_updated (user_id, updated_at),
                CONSTRAINT fk_conversation_sessions_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='user conversation sessions'
            """
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT 'message id',
                session_id VARCHAR(32) NOT NULL COMMENT 'conversation id',
                user_id INT NOT NULL COMMENT 'owner user id',
                question MEDIUMTEXT NOT NULL COMMENT 'user question',
                answer MEDIUMTEXT NOT NULL COMMENT 'assistant answer',
                created_at VARCHAR(32) NOT NULL COMMENT 'created time',
                INDEX idx_conversation_messages_user_session (user_id, session_id, id),
                CONSTRAINT fk_conversation_messages_session_id FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
                CONSTRAINT fk_conversation_messages_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='conversation message rounds'
            """
        )
        try:
            self._execute("ALTER TABLE conversation_sessions CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            self._execute("ALTER TABLE conversation_messages CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            column = self._execute(
                """
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'conversation_sessions'
                  AND COLUMN_NAME = 'session_type'
                """,
                fetchone=True,
            )
            if not column:
                self._execute("ALTER TABLE conversation_sessions ADD COLUMN session_type VARCHAR(16) DEFAULT 'chat' AFTER title")
        except Exception:
            pass

    def new_session_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def list_sessions(self, user_id: int) -> List[dict]:
        sessions = {}
        
        try:
            rows = self._execute(
                """
                SELECT session_id, title AS last_question, session_type, created_at, updated_at
                FROM conversation_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                [user_id],
                fetchall=True,
            )
        except Exception:
            rows = self._execute(
                """
                SELECT session_id, title AS last_question, created_at, updated_at
                FROM conversation_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                [user_id],
                fetchall=True,
            )
        for row in rows:
            if 'session_type' not in row:
                row['session_type'] = 'chat'
            sessions[row['session_id']] = self._timestamp_payload(row)
        
        for json_path in self.storage_dir.glob("*.json"):
            session_id = json_path.stem
            if session_id in sessions:
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not data:
                    continue
                json_user_id = data.get("user_id")
                if json_user_id is not None and json_user_id != user_id:
                    continue
                if isinstance(data, dict) and "conversation" in data:
                    messages = data["conversation"]
                else:
                    messages = data
                
                if messages and isinstance(messages, list) and len(messages) > 0:
                    first_msg = messages[0]
                    title = first_msg.get("question", "")[:20] or "New Conversation"
                    created_at = first_msg.get("created_at", datetime.now().isoformat())
                    
                    session_type = "chat"
                    title_bytes = title.encode('utf-8')
                    if b'\xe6\xa8\xa1\xe6\x8b\x9f' in title_bytes or b'\xe9\x9d\xa2\xe8\xaf\x95' in title_bytes:
                        session_type = "interview"
                    elif b'\xe9\x92\x88\xe5\xaf\xb9' in title_bytes or b'\xe5\x8d\x95\xe4\xbd\x8d' in title_bytes:
                        session_type = "job"
                    
                    sessions[session_id] = {
                        "session_id": session_id,
                        "last_question": title,
                        "session_type": session_type,
                        "created_at": datetime.fromisoformat(created_at).timestamp() if isinstance(created_at, str) else created_at,
                        "updated_at": datetime.fromisoformat(created_at).timestamp() if isinstance(created_at, str) else created_at,
                    }
            except Exception:
                pass
        
        return sorted(sessions.values(), key=lambda x: x.get("updated_at", 0), reverse=True)

    def assert_owner(self, user_id: int, session_id: str) -> None:
        row = self._execute(
            "SELECT session_id FROM conversation_sessions WHERE session_id = ? AND user_id = ?",
            [session_id, user_id],
            fetchone=True,
        )
        if not row:
            raise PermissionError("No permission to access this conversation")

    def get(self, session_id: str, user_id: Optional[int] = None) -> List[dict]:
        if user_id is None:
            return self._get_archived_json(session_id)
        self.assert_owner(user_id, session_id)
        key = (user_id, session_id)
        with self._lock:
            if not self._sessions[key]:
                rows = self._execute(
                    """
                    SELECT question, answer, created_at
                    FROM conversation_messages
                    WHERE session_id = ? AND user_id = ?
                    ORDER BY id ASC
                    """,
                    [session_id, user_id],
                    fetchall=True,
                )
                self._sessions[key] = deque(rows)
            return list(self._sessions[key])

    def append(self, session_id: str, question: str, answer: str, user_id: int, session_type: str = 'chat', title: str = None) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        if title is None:
            title = self._title(question)
        with self._lock:
            existing = self._execute(
                "SELECT session_id FROM conversation_sessions WHERE session_id = ? AND user_id = ?",
                [session_id, user_id],
                fetchone=True,
            )
            if not existing:
                occupied = self._execute(
                    "SELECT user_id FROM conversation_sessions WHERE session_id = ?",
                    [session_id],
                    fetchone=True,
                )
                if occupied:
                    raise PermissionError("No permission to access this conversation")
                self._execute(
                    "INSERT INTO conversation_sessions (session_id, user_id, title, session_type, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    [session_id, user_id, title, session_type, now, now],
                )
            else:
                self._execute(
                    "UPDATE conversation_sessions SET session_type = ?, updated_at = ? WHERE session_id = ? AND user_id = ?",
                    [session_type, now, session_id, user_id],
                )
            self._execute(
                "INSERT INTO conversation_messages (session_id, user_id, question, answer, created_at) VALUES (?, ?, ?, ?, ?)",
                [session_id, user_id, question, answer, now],
            )
            self._sessions[(user_id, session_id)].append({"question": question, "answer": answer, "created_at": now})

    def clear(self, session_id: str, user_id: int) -> None:
        self.assert_owner(user_id, session_id)
        with self._lock:
            self._execute("DELETE FROM conversation_messages WHERE session_id = ? AND user_id = ?", [session_id, user_id])
            self._execute("DELETE FROM conversation_sessions WHERE session_id = ? AND user_id = ?", [session_id, user_id])
            self._sessions.pop((user_id, session_id), None)

    def rename_session(self, session_id: str, user_id: int, title: str) -> None:
        self.assert_owner(user_id, session_id)
        with self._lock:
            self._execute("UPDATE conversation_sessions SET title = ? WHERE session_id = ? AND user_id = ?", [title, session_id, user_id])

    def clear_user(self, user_id: int) -> None:
        with self._lock:
            self._execute("DELETE FROM conversation_messages WHERE user_id = ?", [user_id])
            self._execute("DELETE FROM conversation_sessions WHERE user_id = ?", [user_id])
            for key in [key for key in self._sessions if key[0] == user_id]:
                self._sessions.pop(key, None)

    def format_context(self, session_id: str, user_id: int) -> str:
        try:
            rounds = self.get(session_id, user_id)[-self.max_rounds :]
        except PermissionError:
            return "None"
        if not rounds:
            return "None"
        lines = []
        for index, item in enumerate(rounds, start=1):
            lines.append(f"Round {index} User: {item['question']}")
            lines.append(f"Round {index} Assistant: {item['answer'][:500]}")
        return "\n".join(lines)

    @staticmethod
    def _title(question: str) -> str:
        text = " ".join((question or "").split())
        return text[:20] or "New Conversation"

    @staticmethod
    def _timestamp_payload(row: dict) -> dict:
        payload = dict(row)
        for key in ("created_at", "updated_at"):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                continue
            try:
                payload[key] = datetime.fromisoformat(str(value)).timestamp()
            except Exception:
                payload[key] = 0
        return payload

    def _get_archived_json(self, session_id: str) -> List[dict]:
        try:
            path = self.storage_dir / f"{session_id}.json"
            if not path.exists():
                return []
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("conversation", [])
        except Exception:
            return []

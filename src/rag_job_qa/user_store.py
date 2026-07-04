from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from .config import Settings


@dataclass
class User:
    id: int
    username: str
    display_name: str
    email: str = ""
    phone: str = ""
    avatar: str = ""


class UserStore:
    """User data access layer backed by MySQL."""

    def __init__(self, settings: Settings):
        self.settings = settings
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
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(query, tuple(params))
            if fetchone:
                row = cur.fetchone()
                return dict(row) if row else None
            if fetchall:
                rows = cur.fetchall()
                return [dict(row) for row in rows]
            if query.lstrip().upper().startswith("INSERT"):
                return cur.lastrowid
            return cur.rowcount

    def _init_schema(self) -> None:
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'user primary key',
                username VARCHAR(64) NOT NULL UNIQUE COMMENT 'login username',
                email VARCHAR(128) DEFAULT NULL COMMENT 'email for login',
                phone VARCHAR(32) DEFAULT NULL COMMENT 'phone for login',
                display_name VARCHAR(64) NOT NULL COMMENT 'display name',
                password_hash VARCHAR(256) NOT NULL COMMENT 'PBKDF2 salted password hash',
                created_at VARCHAR(32) NOT NULL COMMENT 'creation time'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='system users table'
            """
        )
        self._ensure_mysql_user_columns()
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS auth_tokens (
                token VARCHAR(64) PRIMARY KEY COMMENT 'login token',
                user_id INT NOT NULL COMMENT 'associated user ID',
                created_at VARCHAR(32) NOT NULL COMMENT 'token creation time',
                INDEX idx_auth_tokens_user_id (user_id),
                CONSTRAINT fk_auth_tokens_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='auth tokens table'
            """
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS user_goals (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'goal primary key',
                user_id INT NOT NULL COMMENT 'associated user ID',
                title VARCHAR(128) NOT NULL COMMENT 'goal title',
                date VARCHAR(32) NOT NULL COMMENT 'target date',
                created_at VARCHAR(32) NOT NULL COMMENT 'creation time',
                INDEX idx_user_goals_user_id (user_id),
                CONSTRAINT fk_user_goals_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='user goals table'
            """
        )

    def _ensure_mysql_user_columns(self) -> None:
        columns = self._execute("SHOW COLUMNS FROM users", fetchall=True)
        names = {item["Field"] for item in columns}
        if "email" not in names:
            self._execute("ALTER TABLE users ADD COLUMN email VARCHAR(128) DEFAULT NULL COMMENT 'email' AFTER username")
        if "phone" not in names:
            self._execute("ALTER TABLE users ADD COLUMN phone VARCHAR(32) DEFAULT NULL COMMENT 'phone' AFTER email")
        if "avatar" not in names:
            self._execute("ALTER TABLE users ADD COLUMN avatar VARCHAR(256) DEFAULT NULL COMMENT 'avatar file path' AFTER phone")
        indexes = self._execute("SHOW INDEX FROM users", fetchall=True)
        index_names = {item["Key_name"] for item in indexes}
        if "idx_users_email" not in index_names:
            self._execute("CREATE INDEX idx_users_email ON users(email)")
        if "idx_users_phone" not in index_names:
            self._execute("CREATE INDEX idx_users_phone ON users(phone)")

    @staticmethod
    def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
        salt = salt or os.urandom(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
        return f"pbkdf2_sha256${salt.hex()}${digest.hex()}"

    @staticmethod
    def _verify_password(password: str, stored: str) -> bool:
        try:
            _, salt_hex, expected = stored.split("$", 2)
        except ValueError:
            return False
        actual = UserStore._hash_password(password, bytes.fromhex(salt_hex)).split("$", 2)[2]
        return hmac.compare_digest(actual, expected)

    @staticmethod
    def _user_from_row(row: Dict[str, Any]) -> User:
        return User(
            id=int(row["id"]),
            username=str(row["username"]),
            display_name=str(row["display_name"]),
            email=str(row.get("email") or ""),
            phone=str(row.get("phone") or ""),
            avatar=str(row.get("avatar") or ""),
        )

    def register(self, username: str, password: str, display_name: str = "", email: str = "", phone: str = "") -> User:
        username = username.strip()
        email = email.strip()
        phone = phone.strip()
        display_name = display_name.strip() or username
        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters")
        exists = self._execute("SELECT id FROM users WHERE username = ?", [username], fetchone=True)
        if exists:
            raise ValueError("Username already exists")
        if email and self._execute("SELECT id FROM users WHERE email = ?", [email], fetchone=True):
            raise ValueError("Email already registered")
        if phone and self._execute("SELECT id FROM users WHERE phone = ?", [phone], fetchone=True):
            raise ValueError("Phone already registered")
        password_hash = self._hash_password(password)
        self._execute(
            "INSERT INTO users (username, email, phone, display_name, password_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            [username, email or None, phone or None, display_name, password_hash, datetime.now().isoformat(timespec="seconds")],
        )
        row = self._execute("SELECT id, username, email, phone, display_name, avatar FROM users WHERE username = ?", [username], fetchone=True)
        return self._user_from_row(row)

    def login(self, username: str, password: str) -> tuple[User, str]:
        account = username.strip()
        row = self._execute(
            "SELECT * FROM users WHERE username = ? OR email = ? OR phone = ?",
            [account, account, account],
            fetchone=True,
        )
        if not row or not self._verify_password(password, str(row["password_hash"])):
            raise ValueError("Invalid username or password")
        token = uuid.uuid4().hex
        self._execute(
            "INSERT INTO auth_tokens (token, user_id, created_at) VALUES (?, ?, ?)",
            [token, int(row["id"]), datetime.now().isoformat(timespec="seconds")],
        )
        return self._user_from_row(row), token

    def get_user_by_token(self, token: str) -> Optional[User]:
        if not token:
            return None
        row = self._execute(
            """
            SELECT u.id, u.username, u.email, u.phone, u.display_name, u.avatar
            FROM auth_tokens t
            JOIN users u ON u.id = t.user_id
            WHERE t.token = ?
            """,
            [token],
            fetchone=True,
        )
        return self._user_from_row(row) if row else None

    def logout(self, token: str) -> None:
        if token:
            self._execute("DELETE FROM auth_tokens WHERE token = ?", [token])

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        row = self._execute(
            "SELECT id, username, email, phone, display_name, avatar FROM users WHERE id = ?",
            [user_id],
            fetchone=True,
        )
        return self._user_from_row(row) if row else None

    def update_user(self, user_id: int, display_name: str = "", email: str = "", phone: str = "") -> None:
        updates = []
        params = []
        if display_name:
            updates.append("display_name = ?")
            params.append(display_name)
        if email:
            updates.append("email = ?")
            params.append(email)
        if phone:
            updates.append("phone = ?")
            params.append(phone)
        if not updates:
            return
        params.append(user_id)
        self._execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)

    def update_avatar(self, user_id: int, avatar: str) -> None:
        self._execute("UPDATE users SET avatar = ? WHERE id = ?", [avatar, user_id])

    def update_password(self, user_id: int, old_password: str, new_password: str) -> None:
        if len(new_password) < 6:
            raise ValueError("New password must be at least 6 characters")
        row = self._execute("SELECT password_hash FROM users WHERE id = ?", [user_id], fetchone=True)
        if not row or not self._verify_password(old_password, str(row["password_hash"])):
            raise ValueError("Invalid old password")
        password_hash = self._hash_password(new_password)
        self._execute("UPDATE users SET password_hash = ? WHERE id = ?", [password_hash, user_id])

    def get_user_goals(self, user_id: int) -> list[dict]:
        rows = self._execute(
            "SELECT id, title, date, created_at FROM user_goals WHERE user_id = ? ORDER BY date ASC",
            [user_id],
            fetchall=True,
        )
        return rows

    def add_user_goal(self, user_id: int, title: str, date: str) -> dict:
        title = title.strip()
        date = date.strip()
        if not title or not date:
            raise ValueError("Title and date are required")
        goal_id = self._execute(
            "INSERT INTO user_goals (user_id, title, date, created_at) VALUES (?, ?, ?, ?)",
            [user_id, title, date, datetime.now().isoformat(timespec="seconds")],
        )
        row = self._execute(
            "SELECT id, title, date, created_at FROM user_goals WHERE id = ?",
            [goal_id],
            fetchone=True,
        )
        return row if row else {}

    def get_user_goal(self, user_id: int, goal_id: int) -> dict | None:
        row = self._execute(
            "SELECT id, title, date, created_at FROM user_goals WHERE id = ? AND user_id = ?",
            [goal_id, user_id],
            fetchone=True,
        )
        return row

    def update_user_goal(self, user_id: int, goal_id: int, title: str, date: str) -> dict | None:
        title = title.strip()
        date = date.strip()
        if not title or not date:
            raise ValueError("Title and date are required")
        self._execute(
            "UPDATE user_goals SET title = ?, date = ? WHERE id = ? AND user_id = ?",
            [title, date, goal_id, user_id],
        )
        return self.get_user_goal(user_id, goal_id)

    def delete_user_goal(self, user_id: int, goal_id: int) -> bool:
        affected = self._execute(
            "DELETE FROM user_goals WHERE id = ? AND user_id = ?",
            [goal_id, user_id],
        )
        return bool(affected)

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import os
import uuid

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import PROJECT_ROOT
from .rag_service import RAGService
from .text_processing import SUPPORTED_SUFFIXES
from .user_store import User, UserStore


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    top_k: int = Field(default=4, ge=1, le=10, description="Recall count")
    category: str = Field(default="", description="Optional 12-category job filter key")


class AuthRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)
    display_name: str = Field(default="", max_length=64)
    email: str = Field(default="", max_length=128)
    phone: str = Field(default="", max_length=32)


class UserUpdateRequest(BaseModel):
    display_name: str = Field(default="", max_length=64)
    email: str = Field(default="", max_length=128)
    phone: str = Field(default="", max_length=32)


class PasswordUpdateRequest(BaseModel):
    old_password: str = Field(..., min_length=6, max_length=128)
    new_password: str = Field(..., min_length=6, max_length=128)


def ok(data=None, message: str = "success") -> dict:
    return {"code": 0, "message": message, "data": data}


def create_app(service: RAGService) -> FastAPI:
    app = FastAPI(
        title="RAG Student Job QA System",
        description="Student job knowledge QA, document upload, user management and streaming answer API",
        version="1.0.0",
    )
    users = UserStore(service.settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    frontend_dir = PROJECT_ROOT / "frontend"
    if frontend_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dir)), name="assets")

    photo_dir = PROJECT_ROOT / "data" / "photo"
    photo_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/photos", StaticFiles(directory=str(photo_dir)), name="photos")

    def token_from_header(authorization: str = Header(default="")) -> str:
        if authorization.lower().startswith("bearer "):
            return authorization.split(" ", 1)[1].strip()
        return ""

    def current_user(token: str = Depends(token_from_header)) -> User:
        user = users.get_user_by_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Please login first")
        return user

    @app.get("/")
    def index():
        index_file = frontend_dir / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="frontend/index.html not found")
        return FileResponse(index_file)

    @app.get("/health")
    def health():
        return ok({"status": "ok", "knowledge_base": service.knowledge_base.stats()})

    @app.post("/api/auth/register")
    def register(payload: AuthRequest):
        try:
            user = users.register(
                payload.username,
                payload.password,
                payload.display_name,
                payload.email,
                payload.phone,
            )
            user, token = users.login(payload.username, payload.password)
            return ok({"token": token, "user": user.__dict__})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/auth/login")
    def login(payload: AuthRequest):
        try:
            user, token = users.login(payload.username, payload.password)
            return ok({"token": token, "user": user.__dict__})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/auth/me")
    def me(user: User = Depends(current_user)):
        return ok({"user": user.__dict__})

    @app.post("/api/auth/logout")
    def logout(token: str = Depends(token_from_header)):
        users.logout(token)
        return ok(message="logged out")

    @app.put("/api/auth/me")
    def update_profile(payload: UserUpdateRequest, user: User = Depends(current_user)):
        try:
            users.update_user(user.id, payload.display_name, payload.email, payload.phone)
            updated_user = users.get_user_by_id(user.id)
            return ok({"user": updated_user.__dict__})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/auth/avatar")
    async def upload_avatar(file: UploadFile = File(...), user: User = Depends(current_user)):
        try:
            allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
            ext = Path(file.filename or "").suffix.lower()
            if ext not in allowed_extensions:
                raise HTTPException(status_code=400, detail="Only image files are allowed")

            filename = f"avatar_{user.id}_{uuid.uuid4().hex[:8]}{ext}"
            filepath = photo_dir / filename
            filepath.write_bytes(await file.read())

            avatar_path = f"/photos/{filename}"
            users.update_avatar(user.id, avatar_path)
            updated_user = users.get_user_by_id(user.id)
            return ok({"user": updated_user.__dict__})
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/auth/password")
    def update_password(payload: PasswordUpdateRequest, user: User = Depends(current_user)):
        try:
            users.update_password(user.id, payload.old_password, payload.new_password)
            return ok(message="Password updated successfully")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/sessions")
    def list_sessions(user: User = Depends(current_user)):
        sessions = []
        for f in service.memory.storage_dir.glob("*.json"):
            session_id = f.stem
            history = service.memory.get(session_id)
            if history:
                sessions.append({
                    "session_id": session_id,
                    "last_question": history[-1]["question"],
                    "created_at": f.stat().st_ctime
                })
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return ok({"sessions": sessions})

    @app.post("/api/chat")
    def chat(payload: ChatRequest, user: User = Depends(current_user)):
        try:
            result = service.answer(
                question=payload.question,
                session_id=payload.session_id or service.new_session(),
                top_k=payload.top_k,
                category=payload.category,
            )
            return ok(
                {
                    "session_id": result.session_id,
                    "answer": result.answer,
                    "sources": result.source_payload(),
                    "cached": result.cached,
                }
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/chat/stream")
    def chat_stream(payload: ChatRequest, user: User = Depends(current_user)):
        from starlette.responses import StreamingResponse
        import asyncio
        
        async def async_event_stream():
            try:
                for event in service.stream_answer(
                    question=payload.question,
                    session_id=payload.session_id or service.new_session(),
                    top_k=payload.top_k,
                    category=payload.category,
                ):
                    line = json.dumps(event, ensure_ascii=False) + "\n"
                    yield line.encode("utf-8")
                    await asyncio.sleep(0)
            except Exception as exc:
                yield (json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False) + "\n").encode("utf-8")

        return StreamingResponse(async_event_stream(), media_type="application/x-ndjson")

    @app.get("/api/history/{session_id}")
    def history(session_id: str, user: User = Depends(current_user)):
        return ok(service.memory.get(session_id))

    @app.post("/api/session/new")
    def new_session(user: User = Depends(current_user)):
        return ok({"session_id": service.new_session()})

    @app.post("/api/session/{session_id}/clear")
    def clear_session(session_id: str, user: User = Depends(current_user)):
        service.clear_session(session_id)
        return ok(message="cleared")

    @app.get("/api/knowledge/stats")
    def stats(user: User = Depends(current_user)):
        return ok(service.knowledge_base.stats())

    @app.post("/api/knowledge/rebuild")
    def rebuild(user: User = Depends(current_user)):
        return ok(service.knowledge_base.rebuild())

    @app.post("/api/knowledge/upload")
    async def upload(files: List[UploadFile] = File(...), user: User = Depends(current_user)):
        if not files:
            raise HTTPException(status_code=400, detail="Please select files to upload")
        temp_paths: List[Path] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                filename = Path(file.filename or "upload.txt").name
                if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type: {filename}. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
                    )
                content = await file.read()
                if not content:
                    raise HTTPException(status_code=400, detail=f"Empty file is not allowed: {filename}")
                target = Path(temp_dir) / filename
                target.write_bytes(content)
                temp_paths.append(target)
            stats_data = service.knowledge_base.import_files(temp_paths)
        return ok(stats_data)

    @app.get("/api/knowledge/documents")
    def list_documents(
        keyword: str = "",
        category: str = "all",
        page: int = 1,
        page_size: int = 10,
        user: User = Depends(current_user),
    ):
        return ok(service.knowledge_base.list_documents(keyword, page, page_size, category))

    @app.get("/api/knowledge/categories")
    def list_categories(user: User = Depends(current_user)):
        return ok({"categories": service.knowledge_base.categories()})

    @app.get("/api/graph/categories")
    def graph_categories(user: User = Depends(current_user)):
        """Return the 12 job categories with graph/document entity counts."""
        return ok({"categories": service.knowledge_base.graph_categories()})

    @app.get("/api/graph")
    def graph_data(
        category: str = "",
        limit: int = 240,
        user: User = Depends(current_user),
    ):
        """Return graph nodes and edges for the full graph or one job category."""
        return ok(service.knowledge_base.graph_data(category, limit))

    @app.get("/api/graph/node/{node_id}")
    def graph_node_detail(
        node_id: str,
        relation_type: str = "",
        user: User = Depends(current_user),
    ):
        """Return node properties and 1-hop related entities."""
        data = service.knowledge_base.node_detail(node_id, relation_type)
        if not data.get("node"):
            raise HTTPException(status_code=404, detail="Node not found")
        return ok(data)

    @app.get("/api/graph/search")
    def graph_search(
        keyword: str,
        category: str = "",
        limit: int = 30,
        user: User = Depends(current_user),
    ):
        """Fuzzy-search graph nodes, optionally scoped to a job category."""
        return ok({"items": service.knowledge_base.search_graph_nodes(keyword, category, limit)})

    @app.get("/api/knowledge/document/{filename}")
    def read_document(filename: str, user: User = Depends(current_user)):
        try:
            return ok(service.knowledge_base.read_document(filename))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Document not found") from exc

    @app.delete("/api/knowledge/document/{filename}")
    def delete_document(filename: str, user: User = Depends(current_user)):
        service.knowledge_base.delete_document(filename)
        return ok(message="Document deleted")

    return app

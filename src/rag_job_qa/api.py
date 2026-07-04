from __future__ import annotations



import json

import sys

import tempfile

import asyncio

from pathlib import Path

from typing import List, Optional



import os

import uuid



from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import FileResponse, StreamingResponse

from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field



from .config import PROJECT_ROOT

from .rag_service import RAGService

from .speech_service import SpeechService

from .text_processing import SUPPORTED_SUFFIXES

from .user_store import User, UserStore





class ChatRequest(BaseModel):

    question: str = Field(..., min_length=1, description="User question")

    session_id: Optional[str] = Field(default=None, description="Session ID")

    top_k: int = Field(default=4, ge=1, le=10, description="Recall count")

    category: str = Field(default="", description="Optional 12-category job filter key")

    session_type: str = Field(default="chat", description="Session type: chat/interview/job")





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





class GraphNodeRequest(BaseModel):

    label: str = Field(..., description="Neo4j label: JobCategory/Position/Skill/Company/City/Document...")

    properties: dict = Field(default_factory=dict)





class GraphNodeUpdateRequest(BaseModel):

    properties: dict = Field(default_factory=dict)





class GraphRelationRequest(BaseModel):

    source_id: str

    target_id: str

    type: str = Field(..., description="Relationship type, e.g. RELATED_TO/REQUIRES/CONTAINS")

    properties: dict = Field(default_factory=dict)





class BatchDeleteDocumentsRequest(BaseModel):

    filenames: List[str] = Field(default_factory=list, max_items=100)





def ok(data=None, message: str = "success") -> dict:

    return {"code": 0, "message": message, "data": data}





def create_app(service: RAGService) -> FastAPI:

    app = FastAPI(

        title="RAG Student Job QA System",

        description="Student job knowledge QA, document upload, user management and streaming answer API",

        version="1.0.0",

    )

    users = UserStore(service.settings)

    speech = SpeechService(service.settings)



    @app.on_event("shutdown")

    async def shutdown_clients():

        await service.llm.aclose()



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

        return FileResponse(index_file, headers={"Cache-Control": "no-store"})



    @app.get("/favicon.ico")

    def favicon():

        icon_file = frontend_dir / "assets" / "app-icon.svg"

        if not icon_file.exists():

            raise HTTPException(status_code=404, detail="frontend/assets/app-icon.svg not found")

        return FileResponse(icon_file, media_type="image/svg+xml", headers={"Cache-Control": "no-store"})



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

        except ValueError as exc:

            msg = str(exc)

            if "Invalid username or password" in msg:
                msg = "Username or password error"
            raise HTTPException(status_code=400, detail=msg) from exc

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



    @app.get("/api/auth/goals")

    def list_goals(user: User = Depends(current_user)):

        return ok({"goals": users.get_user_goals(user.id)})



    @app.post("/api/auth/goals")

    def add_goal(payload: dict, user: User = Depends(current_user)):

        try:

            goal = users.add_user_goal(user.id, payload.get("title", ""), payload.get("date", ""))

            return ok(goal)

        except Exception as exc:

            raise HTTPException(status_code=400, detail=str(exc)) from exc



    @app.get("/api/auth/goals/{goal_id}")

    def get_goal(goal_id: int, user: User = Depends(current_user)):

        try:

            goal = users.get_user_goal(user.id, goal_id)

            if not goal:

                raise HTTPException(status_code=404, detail="Goal not found")

            return ok(goal)

        except HTTPException:

            raise

        except Exception as exc:

            raise HTTPException(status_code=400, detail=str(exc)) from exc



    @app.put("/api/auth/goals/{goal_id}")

    def update_goal(goal_id: int, payload: dict, user: User = Depends(current_user)):

        try:

            goal = users.update_user_goal(user.id, goal_id, payload.get("title", ""), payload.get("date", ""))

            if not goal:

                raise HTTPException(status_code=404, detail="Goal not found")

            return ok(goal)

        except HTTPException:

            raise

        except Exception as exc:

            raise HTTPException(status_code=400, detail=str(exc)) from exc



    @app.delete("/api/auth/goals/{goal_id}")
    def delete_goal(goal_id: int, user: User = Depends(current_user)):
        try:
            deleted = users.delete_user_goal(user.id, goal_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Goal not found")
            return ok({"deleted": True, "goal_id": goal_id})
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


    @app.get("/api/sessions")

    def list_sessions(user: User = Depends(current_user)):

        return ok({"sessions": service.memory.list_sessions(user.id)})



    @app.post("/api/chat")

    async def chat(payload: ChatRequest, user: User = Depends(current_user)):

        try:

            if payload.session_id:

                service.memory.assert_owner(user.id, payload.session_id)

            result = await service.aanswer(

                question=payload.question,

                user_id=user.id,

                session_id=payload.session_id or service.new_session(),

                top_k=payload.top_k,

                category=payload.category,

                session_type=payload.session_type,

            )

            return ok(

                {

                    "session_id": result.session_id,

                    "answer": result.answer,

                    "sources": result.source_payload(),

                    "cached": result.cached,

                }

            )

        except PermissionError as exc:

            raise HTTPException(status_code=403, detail=str(exc)) from exc

        except Exception as exc:

            raise HTTPException(status_code=400, detail=str(exc)) from exc



    @app.post("/api/chat/stream")

    def chat_stream(payload: ChatRequest, user: User = Depends(current_user)):

        from starlette.responses import StreamingResponse

        import asyncio

        

        if payload.session_id:

            try:

                service.memory.assert_owner(user.id, payload.session_id)

            except PermissionError as exc:

                raise HTTPException(status_code=403, detail=str(exc)) from exc



        async def async_event_stream():

            try:

                async for event in service.astream_answer(

                    question=payload.question,

                    user_id=user.id,

                    session_id=payload.session_id or service.new_session(),

                    top_k=payload.top_k,

                    category=payload.category,

                    session_type=payload.session_type,

                ):

                    line = json.dumps(event, ensure_ascii=False) + "\n"

                    yield line.encode("utf-8")

                    await asyncio.sleep(0.01)

            except Exception as exc:

                yield (json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False) + "\n").encode("utf-8")



        return StreamingResponse(

            async_event_stream(), 

            media_type="application/x-ndjson",

            headers={

                "Cache-Control": "no-cache",

                "Connection": "keep-alive",

                "X-Accel-Buffering": "no",

            }

        )



    @app.get("/api/history/{session_id}")

    def history(session_id: str, user: User = Depends(current_user)):

        try:

            return ok(service.memory.get(session_id, user.id))

        except PermissionError as exc:

            raise HTTPException(status_code=403, detail=str(exc)) from exc



    @app.post("/api/session/new")

    def new_session(user: User = Depends(current_user)):

        return ok({"session_id": service.new_session()})



    @app.post("/api/session/{session_id}/clear")

    def clear_session(session_id: str, user: User = Depends(current_user)):

        try:

            service.clear_session(session_id, user.id)

        except PermissionError as exc:

            raise HTTPException(status_code=403, detail=str(exc)) from exc

        return ok(message="cleared")



    @app.post("/api/session/{session_id}/rename")

    def rename_session(session_id: str, payload: dict, user: User = Depends(current_user)):

        try:

            service.memory.rename_session(session_id, user.id, payload.get("title", ""))

        except PermissionError as exc:

            raise HTTPException(status_code=403, detail=str(exc)) from exc

        return ok(message="renamed")



    @app.delete("/api/sessions")

    def clear_all_sessions(user: User = Depends(current_user)):

        service.memory.clear_user(user.id)

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

        max_bytes = service.settings.max_upload_size_mb * 1024 * 1024

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

                if len(content) > max_bytes:

                    raise HTTPException(

                        status_code=413,

                        detail=f"File is too large: {filename}. Max size is {service.settings.max_upload_size_mb}MB",

                    )

                target = Path(temp_dir) / filename

                target.write_bytes(content)

                temp_paths.append(target)

            stats_data = await asyncio.to_thread(service.knowledge_base.import_files, temp_paths)

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



    @app.get("/api/graph/diagnostics")

    def graph_diagnostics(user: User = Depends(current_user)):

        """Return Neo4j connectivity diagnostics for troubleshooting graph startup."""

        return ok(service.knowledge_base.graph_diagnostics())



    @app.get("/api/graph")

    def graph_data(

        category: str = "",

        limit: int = 120,

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



    @app.post("/api/graph/node")

    def create_graph_node(payload: GraphNodeRequest, user: User = Depends(current_user)):

        try:

            return ok(service.knowledge_base.graph_store.create_node(payload.label, payload.properties))

        except Exception as exc:

            raise HTTPException(status_code=400, detail=str(exc)) from exc



    @app.put("/api/graph/node/{node_id}")

    def update_graph_node(node_id: str, payload: GraphNodeUpdateRequest, user: User = Depends(current_user)):

        try:

            return ok(service.knowledge_base.graph_store.update_node(node_id, payload.properties))

        except ValueError as exc:

            raise HTTPException(status_code=404, detail=str(exc)) from exc

        except Exception as exc:

            raise HTTPException(status_code=400, detail=str(exc)) from exc



    @app.delete("/api/graph/node/{node_id}")

    def delete_graph_node(node_id: str, user: User = Depends(current_user)):

        try:

            service.knowledge_base.graph_store.delete_node(node_id)

            return ok(message="Node deleted")

        except ValueError as exc:

            raise HTTPException(status_code=404, detail=str(exc)) from exc

        except Exception as exc:

            raise HTTPException(status_code=400, detail=str(exc)) from exc



    @app.post("/api/graph/relation")

    def create_graph_relation(payload: GraphRelationRequest, user: User = Depends(current_user)):

        try:

            return ok(service.knowledge_base.graph_store.create_relation(payload.source_id, payload.target_id, payload.type, payload.properties))

        except ValueError as exc:

            raise HTTPException(status_code=404, detail=str(exc)) from exc

        except Exception as exc:

            raise HTTPException(status_code=400, detail=str(exc)) from exc



    @app.delete("/api/graph/relation/{relation_id}")

    def delete_graph_relation(relation_id: str, user: User = Depends(current_user)):

        try:

            service.knowledge_base.graph_store.delete_relation(relation_id)

            return ok(message="Relation deleted")

        except ValueError as exc:

            raise HTTPException(status_code=404, detail=str(exc)) from exc

        except Exception as exc:

            raise HTTPException(status_code=400, detail=str(exc)) from exc



    @app.post("/api/speech/recognize")

    async def recognize_speech(file: UploadFile = File(...), user: User = Depends(current_user)):

        try:

            content = await file.read()

            suffix = Path(file.filename or ".webm").suffix or ".webm"

            return ok(speech.recognize_bytes(content, suffix))

        except Exception as exc:

            raise HTTPException(status_code=400, detail=str(exc)) from exc



    @app.websocket("/ws/speech")

    async def speech_websocket(websocket: WebSocket, token: str = "", suffix: str = ".webm"):

        user = users.get_user_by_token(token)

        if not user:

            await websocket.close(code=1008, reason="Please login first")

            return

        await websocket.accept()

        buffer = bytearray()

        last_emit = 0.0

        last_text = ""

        min_interval = 1.2

        max_buffer_bytes = 8 * 1024 * 1024

        try:

            await websocket.send_json({"type": "ready"})

            while True:

                message = await websocket.receive()

                if message.get("bytes") is not None:

                    chunk = message["bytes"]

                    if chunk:

                        buffer.extend(chunk)

                        if len(buffer) > max_buffer_bytes:

                            buffer = buffer[-max_buffer_bytes:]

                    now = asyncio.get_running_loop().time()

                    if buffer and now - last_emit >= min_interval:

                        try:

                            result = await asyncio.to_thread(speech.recognize_stream_bytes, bytes(buffer), suffix)

                            text = result.get("text", "")

                            if text and text != last_text:

                                last_text = text

                                await websocket.send_json({"type": "partial", "text": text})

                        except Exception as exc:

                            await websocket.send_json({"type": "error", "message": str(exc)})

                        last_emit = now

                elif message.get("text") is not None:

                    try:

                        payload = json.loads(message["text"])

                    except Exception:

                        payload = {"type": message["text"]}

                    if payload.get("type") == "stop":

                        if buffer:

                            result = await asyncio.to_thread(speech.recognize_stream_bytes, bytes(buffer), suffix)

                            last_text = result.get("text", "") or last_text

                        await websocket.send_json({"type": "final", "text": last_text})

                        await websocket.close()

                        return

        except WebSocketDisconnect:

            return

        except Exception as exc:

            try:

                await websocket.send_json({"type": "error", "message": str(exc)})

                await websocket.close(code=1011)

            except Exception:

                pass



    @app.get("/api/knowledge/document/{filename:path}")

    def read_document(filename: str, user: User = Depends(current_user)):

        try:

            return ok(service.knowledge_base.read_document(filename))

        except FileNotFoundError as exc:

            raise HTTPException(status_code=404, detail="Document not found") from exc



    @app.get("/api/knowledge/document-file/{filename:path}")

    def read_document_file(filename: str, token: str = ""):

        if not users.get_user_by_token(token):

            raise HTTPException(status_code=401, detail="Please login first")

        filepath = service.knowledge_base.document_file_path(filename)

        if not filepath.exists() or not filepath.is_file():

            raise HTTPException(status_code=404, detail="Document not found")

        from fastapi.responses import FileResponse

        return FileResponse(filepath, media_type='application/pdf', headers={'Content-Disposition': 'inline'})



    @app.delete("/api/knowledge/document/{filename:path}")

    def delete_document(filename: str, user: User = Depends(current_user)):

        service.knowledge_base.delete_document(filename)

        return ok(message="Document deleted")



    @app.post("/api/knowledge/documents/delete")

    def delete_documents(payload: BatchDeleteDocumentsRequest, user: User = Depends(current_user)):

        filenames = [item for item in payload.filenames if str(item or "").strip()]

        if not filenames:

            raise HTTPException(status_code=400, detail="Please select documents to delete")

        return ok(service.knowledge_base.delete_documents(filenames), message="Documents deleted")



    return app


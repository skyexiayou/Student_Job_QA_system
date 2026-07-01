from __future__ import annotations

import argparse

import uvicorn

from src.rag_job_qa.api import create_app
from src.rag_job_qa.config import Settings
from src.rag_job_qa.rag_service import RAGService


def parse_args():
    parser = argparse.ArgumentParser(description="基础 RAG 大学生求职岗位知识问答系统")
    parser.add_argument("--host", default=None, help="服务监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="FastAPI 服务端口，默认 8000")
    parser.add_argument("--reload", action="store_true", help="开发模式：代码修改后自动重载")
    return parser.parse_args()


def build_app():
    settings = Settings.load()
    settings.ensure_dirs()
    service = RAGService(settings)
    return create_app(service)


app = build_app()


def main() -> None:
    args = parse_args()
    settings = Settings.load()
    host = args.host or settings.api_host
    port = args.port or settings.api_port
    print(f"FastAPI + Vue 页面：http://{host}:{port}")
    print(f"接口文档：http://{host}:{port}/docs")
    uvicorn.run("run:app", host=host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()

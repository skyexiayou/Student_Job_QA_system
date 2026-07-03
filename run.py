from __future__ import annotations

import argparse
import logging
import os
import subprocess

import uvicorn

from src.rag_job_qa.api import create_app
from src.rag_job_qa.config import Settings
from src.rag_job_qa.rag_service import RAGService


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Student Job QA System")
    parser.add_argument("--host", default=None, help="API listen host, default comes from .env")
    parser.add_argument("--port", type=int, default=None, help="API port, default is 8000")
    parser.add_argument("--reload", action="store_true", help="Reload server when code changes")
    parser.add_argument(
        "--no-kill-port",
        action="store_true",
        help="Do not stop an existing process on the target port before startup",
    )
    return parser.parse_args()


def build_app():
    settings = Settings.load()
    settings.ensure_dirs()
    service = RAGService(settings)
    return create_app(service)


app = build_app()


def kill_processes_on_port(port: int) -> list[int]:
    command = (
        "$ErrorActionPreference='SilentlyContinue'; "
        f"Get-NetTCPConnection -LocalPort {int(port)} -State Listen | "
        "Select-Object -ExpandProperty OwningProcess -Unique"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    killed: list[int] = []
    current_pid = os.getpid()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.isdigit():
            continue
        pid = int(line)
        if pid == current_pid:
            continue
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force"],
            capture_output=True,
            text=True,
            check=False,
        )
        killed.append(pid)
    return killed


def main() -> None:
    args = parse_args()
    settings = Settings.load()
    host = args.host or settings.api_host
    port = args.port or settings.api_port or 8000
    if not args.no_kill_port:
        killed = kill_processes_on_port(port)
        if killed:
            print(f"Stopped existing process(es) on port {port}: {', '.join(map(str, killed))}")
    print(f"FastAPI + Vue page: http://{host}:{port}")
    print(f"Swagger docs: http://{host}:{port}/docs")
    uvicorn.run("run:app", host=host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()

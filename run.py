from __future__ import annotations

import argparse
import ipaddress
import logging
import os
import socket
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
    parser = argparse.ArgumentParser(description="RAG 求职岗位知识问答系统")
    parser.add_argument("--host", default=None, help="API 监听地址，默认读取 .env 配置")
    parser.add_argument("--port", type=int, default=None, help="API 端口，默认 8000")
    parser.add_argument("--reload", action="store_true", help="代码修改后自动重启服务")
    parser.add_argument(
        "--no-kill-port",
        action="store_true",
        help="启动前不关闭占用端口的进程",
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


def is_lan_candidate(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    # 过滤基准测试网段（Meta虚拟网卡）
    benchmarking_net = ipaddress.ip_network("198.18.0.0/15")
    # 过滤常见虚拟机虚拟网段（VMware、WSL、Hyper-V）
    virtual_nets = [
        ipaddress.ip_network("192.168.93.0/24"),
        ipaddress.ip_network("192.168.88.0/24"),
        ipaddress.ip_network("172.22.0.0/16"),
        ipaddress.ip_network("172.25.0.0/16"),
        ipaddress.ip_network("172.29.0.0/16"),
        ipaddress.ip_network("192.168.195.0/24"),
    ]
    if any(ip in net for net in virtual_nets):
        return False
    return (
        ip.version == 4
        and ip.is_private
        and not ip.is_loopback
        and not ip.is_link_local
        and ip not in benchmarking_net
    )


def get_physical_lan_ip() -> str:
    """仅获取物理网卡的局域网IP（同WiFi同伴可直接访问），获取失败返回127.0.0.1"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            host = sock.getsockname()[0]
            if is_lan_candidate(host):
                return host
    except OSError:
        pass
    # 兜底遍历所有网卡，取第一个符合条件的物理网卡IP
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            host = info[4][0]
            if is_lan_candidate(host):
                return host
    except OSError:
        pass
    return "127.0.0.1"


def main() -> None:
    args = parse_args()
    settings = Settings.load()
    host = args.host or settings.api_host or "0.0.0.0"
    port = args.port or settings.api_port or 8000

    if not args.no_kill_port:
        killed = kill_processes_on_port(port)
        if killed:
            print(f"已停止端口 {port} 上的现有进程：{', '.join(map(str, killed))}")

    if host in {"0.0.0.0", "::"}:
        lan_ip = get_physical_lan_ip()
        print(f"系统面板：http://{lan_ip}:{port}")
        print(f"接口文档：http://{lan_ip}:{port}/docs")
        print(f"同局域网访问地址：http://{lan_ip}:{port}")
    else:
        print(f"FastAPI + Vue 前端页面：http://{host}:{port}")
        print(f"Swagger 接口文档：http://{host}:{port}/docs")

    uvicorn.run("run:app", host=host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()
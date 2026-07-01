from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 直接运行 scripts/download_hf_model.py 时，Python 默认只把 scripts 加入导入路径。
# 手动补上项目根目录，确保可以导入 src.rag_job_qa。
PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from src.rag_job_qa.config import MODEL_CACHE_DIR, PROJECT_ROOT, Settings


def prepare_hf_env() -> None:
    """把 Hugging Face 缓存固定到项目目录，避免默认写入 C 盘用户目录。"""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(MODEL_CACHE_DIR / "huggingface")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_CACHE_DIR / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_DIR / "huggingface" / "transformers")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def parse_args():
    parser = argparse.ArgumentParser(description="下载 Hugging Face Embedding 模型到项目目录")
    parser.add_argument("--model", default=None, help="模型名称，默认读取 .env 或系统默认配置")
    parser.add_argument(
        "--manual-dir",
        default=None,
        help="可选：直接下载到指定本地文件夹，便于在 .env 中把 EMBEDDING_MODEL 指向该目录",
    )
    return parser.parse_args()


def main() -> None:
    prepare_hf_env()
    args = parse_args()
    settings = Settings.load()
    model_name = args.model or settings.embedding_model

    # 环境变量必须在导入 sentence_transformers / huggingface_hub 前设置。
    from sentence_transformers import SentenceTransformer

    print(f"正在下载 Hugging Face Embedding 模型：{model_name}")
    print(f"HF_HOME：{os.environ['HF_HOME']}")

    if args.manual_dir:
        target = Path(args.manual_dir)
        if not target.is_absolute():
            target = PROJECT_ROOT / target
        target.mkdir(parents=True, exist_ok=True)
        print(f"本地模型目录：{target}")
        SentenceTransformer(model_name, cache_folder=str(target), local_files_only=False).save(str(target))
        print("模型已保存到本地目录。请在 .env 中设置 EMBEDDING_MODEL 为该目录的绝对路径。")
        return

    SentenceTransformer(model_name, local_files_only=False)
    print("模型下载完成。之后系统会自动启用 Hugging Face Embedding。")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_CACHE_DIR = DATA_DIR / "models"

os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE_DIR / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE_DIR / "huggingface" / "transformers"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_dotenv(path: Path) -> Dict[str, str]:
    """Read a simple .env file without requiring python-dotenv."""
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def default_api_key_dir() -> Path:
    """Locate the sibling API Key folder used in the internship workspace."""
    internship_root = PROJECT_ROOT.parents[1] if len(PROJECT_ROOT.parents) > 1 else PROJECT_ROOT
    return internship_root / "API Key"


def _read_key_value_csv(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file):
            if len(row) >= 2 and row[0].strip():
                values[row[0].strip()] = row[1].strip()
    return values


def load_api_csv(api_key_dir: Optional[Path] = None) -> Dict[str, str]:
    """Read the newest API Key CSV without logging secret values."""
    folders = [api_key_dir or Path(os.getenv("API_KEY_DIR", default_api_key_dir())), DATA_DIR / "api_key"]
    csv_files = []
    for folder in folders:
        if folder.exists():
            csv_files.extend(folder.glob("*.csv"))
    if not csv_files:
        return {}
    csv_files = sorted(csv_files, key=lambda item: item.stat().st_mtime, reverse=True)
    return _read_key_value_csv(csv_files[0])


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    documents_dir: Path = DATA_DIR / "knowledge_base" / "uploads"
    legacy_documents_dir: Path = DATA_DIR / "documents"
    seed_knowledge_dir: Path = DATA_DIR / "knowledge_base"
    index_dir: Path = DATA_DIR / "index"
    conversations_dir: Path = DATA_DIR / "conversations"
    model_cache_dir: Path = MODEL_CACHE_DIR

    api_key: str = ""
    base_url: str = ""
    model_name: str = "qwen-plus"
    request_timeout: int = 60
    llm_max_tokens: int = 650
    allow_llm_fallback: bool = True

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_hf_embedding: bool = True
    hf_local_files_only: bool = True
    chunk_size: int = 650
    chunk_overlap: int = 100
    default_top_k: int = 4
    similarity_threshold: float = 0.0
    graph_hops: int = 2
    max_upload_size_mb: int = 30
    memory_rounds: int = 4
    cache_size: int = 128

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "rag_job_qa"

    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    neo4j_vector_index: str = "job_chunk_embedding_index"
    neo4j_vector_dimension: int = 384
    funasr_model: str = "iic/SenseVoiceSmall"

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")
        csv_values = load_api_csv()

        api_key = (
            os.getenv("QWEN_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or csv_values.get("apiKey", "")
        )
        base_url = (
            os.getenv("QWEN_BASE_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or csv_values.get("openAiCompatible", "")
            or csv_values.get("apiHost", "")
        )

        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model_name=os.getenv("QWEN_MODEL", os.getenv("OPENAI_MODEL", "qwen-plus")),
            request_timeout=int(os.getenv("LLM_TIMEOUT", "60")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "650")),
            allow_llm_fallback=os.getenv("ALLOW_LLM_FALLBACK", "true").lower() != "false",
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            use_hf_embedding=os.getenv("USE_HF_EMBEDDING", "true").lower() == "true",
            hf_local_files_only=os.getenv("HF_LOCAL_FILES_ONLY", "true").lower() == "true",
            chunk_size=int(os.getenv("CHUNK_SIZE", "650")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "4")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0")),
            graph_hops=max(1, min(int(os.getenv("GRAPH_HOPS", "2")), 2)),
            max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "30")),
            memory_rounds=int(os.getenv("MEMORY_ROUNDS", "4")),
            mysql_host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            mysql_port=int(os.getenv("MYSQL_PORT", "3306")),
            mysql_user=os.getenv("MYSQL_USER", "root"),
            mysql_password=os.getenv("MYSQL_PASSWORD", ""),
            mysql_database=os.getenv("MYSQL_DATABASE", "rag_job_qa"),
            neo4j_uri=os.getenv("NEO4J_URI", ""),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            neo4j_vector_index=os.getenv("NEO4J_VECTOR_INDEX", "job_chunk_embedding_index"),
            neo4j_vector_dimension=int(os.getenv("NEO4J_VECTOR_DIMENSION", "384")),
        )

    def ensure_dirs(self) -> None:
        for path in [
            self.data_dir,
            self.documents_dir,
            self.legacy_documents_dir,
            self.seed_knowledge_dir,
            self.index_dir,
            self.conversations_dir,
            self.model_cache_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

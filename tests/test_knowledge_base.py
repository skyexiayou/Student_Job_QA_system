from pathlib import Path

from src.rag_job_qa.config import Settings
from src.rag_job_qa.knowledge_base import KnowledgeBase


def test_default_upload_dir_is_under_knowledge_base():
    settings = Settings()
    assert settings.documents_dir == settings.seed_knowledge_dir / "uploads"


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        documents_dir=tmp_path / "documents",
        seed_knowledge_dir=tmp_path / "knowledge_base",
        index_dir=tmp_path / "index",
        conversations_dir=tmp_path / "conversations",
        model_cache_dir=tmp_path / "models",
        use_hf_embedding=False,
    )


def test_import_files_parses_classifies_and_indexes_uploaded_docs(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    source = tmp_path / "测试工程师岗位说明.txt"
    source.write_text("测试工程师 需要 自动化测试 pytest selenium 测试用例 接口测试", encoding="utf-8")

    kb = KnowledgeBase(settings)
    stats = kb.import_files([source])

    assert stats["imported"] == 1
    assert stats["skipped_parse_failed"] == 0
    assert stats["document_count"] == 1
    assert stats["chunk_count"] >= 1
    assert stats["imported_documents"][0]["category"] == "testing"

    listed = kb.list_documents(category="testing")
    assert listed["total"] == 1
    assert listed["documents"][0]["category"] == "testing"
    assert listed["documents"][0]["chunk_count"] >= 1


def test_document_ids_work_for_nested_upload_directory(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    nested = settings.documents_dir / "2026-07-01"
    nested.mkdir(parents=True)
    doc = nested / "后端开发资料.txt"
    doc.write_text("后端开发 Java Spring MySQL Redis 接口 高并发", encoding="utf-8")

    kb = KnowledgeBase(settings)
    kb.rebuild()

    listed = kb.list_documents(category="backend")
    assert listed["total"] == 1
    doc_id = listed["documents"][0]["id"]
    assert doc_id == "2026-07-01/后端开发资料.txt"
    assert kb.document_file_path(doc_id) == doc

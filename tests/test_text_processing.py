from src.rag_job_qa.text_processing import clean_text, split_text
from src.rag_job_qa.vector_store import EmbeddingBackend


def test_clean_text_removes_extra_blank_lines():
    text = "岗位要求：  Python\n\n\n\n熟悉 Flask\t和 RAG"
    cleaned = clean_text(text)
    assert "\n\n\n" not in cleaned
    assert "Python" in cleaned


def test_split_text_returns_chunks():
    text = "简历要突出项目经历。" * 120
    chunks = split_text(text, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_hashing_embedding_is_stable_across_instances():
    first = EmbeddingBackend("unused", use_hf_embedding=False).encode(["Python 后端开发"])
    second = EmbeddingBackend("unused", use_hf_embedding=False).encode(["Python 后端开发"])
    assert first.shape == second.shape
    assert (first == second).all()

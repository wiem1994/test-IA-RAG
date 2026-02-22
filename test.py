from __future__ import annotations

from pathlib import Path

from rag import SimpleRAG


def main() -> int:
    data_dir = Path(__file__).parent / "data"
    rag = SimpleRAG(data_dir)
    rag.build_index()
    chunks = rag.retrieve("What is RAG?", top_k=2)
    assert chunks, "No chunks retrieved"
    assert any("RAG" in c.text for c in chunks), "Expected RAG in retrieved chunks"
    print("OK: retrieval returns relevant chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

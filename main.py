from __future__ import annotations

import sys
from pathlib import Path

from rag import SimpleRAG


def main() -> int:
    data_dir = Path(__file__).parent / "data"
    rag = SimpleRAG(data_dir)
    question = " ".join(sys.argv[1:]).strip() or "What is RAG and why is retrieval useful?"
    answer, _chunks = rag.answer(question, top_k=3)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


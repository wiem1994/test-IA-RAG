from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    text: str
    source: str


class SimpleRAG:
    def __init__(self, data_dir: str | Path, chunk_size: int = 400, chunk_overlap: int = 80) -> None:
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: List[Chunk] = []
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None

    def load_documents(self) -> List[Tuple[str, str]]:
        docs: List[Tuple[str, str]] = []
        for path in sorted(self.data_dir.glob("*.txt")):
            content = path.read_text(encoding="utf-8")
            docs.append((content, path.name))
        return docs

    def _split_text(self, text: str) -> List[str]:
        # Sentence-like split, then merge into fixed-size chunks.
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: List[str] = []
        current = ""
        for part in parts:
            if not part:
                continue
            if len(current) + len(part) + 1 <= self.chunk_size:
                current = f"{current} {part}".strip()
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        # Add overlap by simple tail prepend.
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped: List[str] = []
            for idx, chunk in enumerate(chunks):
                if idx == 0:
                    overlapped.append(chunk)
                else:
                    tail = chunks[idx - 1][-self.chunk_overlap :]
                    overlapped.append(f"{tail} {chunk}".strip())
            chunks = overlapped
        return chunks

    def build_index(self) -> None:
        self._chunks = []
        for content, source in self.load_documents():
            for chunk in self._split_text(content):
                self._chunks.append(Chunk(text=chunk, source=source))
        texts = [c.text for c in self._chunks]
        if not texts:
            self._matrix = None
            return
        self._matrix = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[Chunk]:
        if self._matrix is None:
            self.build_index()
        if self._matrix is None:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_idx = scores.argsort()[::-1][:top_k]
        return [self._chunks[i] for i in top_idx]

    def answer(self, query: str, top_k: int = 3) -> Tuple[str, List[Chunk]]:
        chunks = self.retrieve(query, top_k=top_k)
        if not chunks:
            return "No data found.", []
        # Basic synthesis: surface top passages and cite sources.
        bullet_points = [f"- {c.text}" for c in chunks]
        sources = ", ".join(sorted({c.source for c in chunks}))
        response = "\n".join(
            [
                f"Question: {query}",
                "Answer (extractive):",
                *bullet_points,
                f"Sources: {sources}",
            ]
        )
        return response, chunks


"""Cutting the source documents into passages that can be retrieved and cited.

Input: the `.txt` text of each source document in `data/safety/docs/` (one "[page N]" block per page).
Output: `Passage`s of roughly `max_chars` characters, cut on paragraph and line boundaries, each
knowing its document and page, so an answer can say "WHO-5K, page 19".
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from foodrescue.config import PROJECT_ROOT

DOCS = PROJECT_ROOT / "data" / "safety" / "docs"
_PAGE = re.compile(r"^\[page (\d+)\][^\n]*\n", re.M)


class Passage(BaseModel):
    id: str  # e.g. "WHO-5K p19 #1"
    doc: str
    page: int
    text: str

    def citation(self) -> str:
        return f"{self.doc}, page {self.page}"


def split_text(text: str, max_chars: int = 800) -> list[str]:
    """Greedy packing of paragraphs (then lines) into chunks of at most ~max_chars."""
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
        else:
            pieces += [line.strip() for line in paragraph.splitlines() if line.strip()]
    chunks, current = [], ""
    for piece in filter(None, pieces):
        if current and len(current) + len(piece) + 1 > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def load_corpus(docs: Path = DOCS, max_chars: int = 800, min_chars: int = 60) -> list[Passage]:
    passages = []
    for path in sorted(docs.glob("*/*.txt")):
        content = path.read_text(encoding="utf-8")
        doc = content.splitlines()[0].removeprefix("# ").strip()
        blocks = _PAGE.split(content)[1:]  # [page, text, page, text, ...]
        for page, text in zip(blocks[::2], blocks[1::2], strict=True):
            for i, chunk in enumerate(split_text(text, max_chars), 1):
                if len(chunk) >= min_chars:
                    passages.append(Passage(id=f"{doc} p{page} #{i}", doc=doc, page=int(page), text=chunk))
    return passages

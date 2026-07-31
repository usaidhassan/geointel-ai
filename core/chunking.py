"""
Chunking for long-form documents (research papers), following the
"Chunking for Longer Texts" approach from the course: each paper gets a
stable doc_id, each retrievable piece gets a chunk_id, and chunks are cut
along section boundaries first (Abstract / Introduction / Methods / ... )
rather than blindly by character count, so a chunk doesn't start mid-sentence.

If a section is still too long for one chunk, it's split further with a
sliding window and a small overlap so context isn't lost at the seam.
"""
import re
from dataclasses import dataclass, field

import fitz  # pymupdf

# Common section headers in research papers (case-insensitive, optionally numbered)
SECTION_HEADER_RE = re.compile(
    r"^\s*(?:\d+[\.\)]?\s*)?"
    r"(abstract|introduction|related work|background|literature review|"
    r"materials and methods|methodology|methods|data|study area|"
    r"results|results and discussion|discussion|conclusion|conclusions|"
    r"acknowledg(e)?ments|references)\s*$",
    re.IGNORECASE,
)

MAX_CHUNK_WORDS = 220
OVERLAP_WORDS = 40


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)


def extract_text_from_pdf(path: str) -> str:
    """Extract raw text from a PDF, page by page, preserving line breaks."""
    doc = fitz.open(path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split raw paper text into (section_name, section_text) pairs using
    common academic section headers as split points. Falls back to a
    single "body" section if no headers are detected (e.g. a report or
    tutorial doc with a different structure).
    """
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_name = "front_matter"
    current_lines: list[str] = []

    for line in lines:
        if SECTION_HEADER_RE.match(line.strip()):
            if current_lines:
                sections.append((current_name, current_lines))
            current_name = line.strip().lower()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_name, current_lines))

    # If we only ever found "front_matter", the header regex didn't match
    # anything in this document -> treat the whole thing as one "body" section.
    if len(sections) == 1 and sections[0][0] == "front_matter":
        return [("body", "\n".join(sections[0][1]))]

    return [(name, "\n".join(ls)) for name, ls in sections]


def _split_long_section(section_text: str) -> list[str]:
    """Sliding-window word split with overlap, used only when a single
    section is longer than MAX_CHUNK_WORDS."""
    words = section_text.split()
    if len(words) <= MAX_CHUNK_WORDS:
        return [section_text.strip()] if section_text.strip() else []

    pieces = []
    start = 0
    step = MAX_CHUNK_WORDS - OVERLAP_WORDS
    while start < len(words):
        piece = " ".join(words[start : start + MAX_CHUNK_WORDS])
        if piece.strip():
            pieces.append(piece.strip())
        start += step
    return pieces


def chunk_document(doc_id: str, text: str, extra_metadata: dict | None = None) -> list[Chunk]:
    """Full pipeline: raw paper text -> list of Chunk objects with stable ids."""
    extra_metadata = extra_metadata or {}
    sections = split_into_sections(text)

    chunks: list[Chunk] = []
    idx = 0
    for section_name, section_text in sections:
        for piece in _split_long_section(section_text):
            if len(piece.split()) < 6:
                # Skip near-empty fragments (e.g. a lone header with no body,
                # or a stray page-break artifact) - not useful for retrieval.
                # Kept deliberately low so short-but-real sections (a brief
                # Results paragraph, a one-line Conclusion) aren't discarded.
                continue
            chunk_id = f"{doc_id}_{idx:03d}"
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    section=section_name,
                    text=piece,
                    metadata=extra_metadata,
                )
            )
            idx += 1
    return chunks


def chunk_pdf(path: str, doc_id: str, extra_metadata: dict | None = None) -> list[Chunk]:
    text = extract_text_from_pdf(path)
    return chunk_document(doc_id, text, extra_metadata)

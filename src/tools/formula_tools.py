import re
from pathlib import Path
from typing import TypedDict


class FormulaChunk(TypedDict):
    section: str
    content: str
    formulas: list[str]


INLINE_FORMULA = re.compile(r"(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)")
BLOCK_FORMULA  = re.compile(r"\$\$.*?\$\$", re.DOTALL)
HEADER_PATTERN = re.compile(r"(?=^#{1,3} .+$)", re.MULTILINE)
HEADER_MATCH   = re.compile(r"^(#{1,3}) (.+)$", re.MULTILINE)


def extract_formulas(text: str) -> list[str]:
    block = BLOCK_FORMULA.findall(text)
    clean_text = BLOCK_FORMULA.sub("", text)
    inline = INLINE_FORMULA.findall(clean_text)
    return [f.strip() for f in block + inline]


def split_into_sections(md_text: str) -> list[dict]:
    sections = []
    for part in HEADER_PATTERN.split(md_text):
        if not part.strip():
            continue
        match = HEADER_MATCH.search(part)
        heading = match.group(2).strip() if match else "Introduction"
        sections.append({"heading": heading, "content": part.strip()})
    return sections


def chunk_section(content: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks, current, current_len = [], [], 0

    for para in paragraphs:
        if current_len + len(para) > chunk_size and current:
            chunks.append("\n\n".join(current))
            overlap_paras, overlap_len = [], 0
            for p in reversed(current):
                if overlap_len + len(p) <= overlap:
                    overlap_paras.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            current, current_len = overlap_paras, overlap_len
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def parse_formula_chunks(md_path: str | Path) -> list[FormulaChunk]:
    with open(Path(md_path), encoding="utf-8") as f:
        md_text = f.read()

    formula_chunks: list[FormulaChunk] = []
    for sec in split_into_sections(md_text):
        for chunk in chunk_section(sec["content"]):
            formulas = extract_formulas(chunk)
            if formulas:
                formula_chunks.append(FormulaChunk(
                    section=sec["heading"],
                    content=chunk,
                    formulas=formulas,
                ))
    return formula_chunks
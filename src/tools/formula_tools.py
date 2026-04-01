import re
from pathlib import Path
from typing import TypedDict, List, Dict, Any
from src.config import model

# Batch boyutunu büyüt — 20-50 formül için 4000 yeterli değil
BATCH_CHARS = 8_000

INLINE_FORMULA = re.compile(r"(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)")
BLOCK_FORMULA  = re.compile(r"\$\$.*?\$\$", re.DOTALL)
HEADER_PATTERN = re.compile(r"(?=^#{1,6} .+$)", re.MULTILINE)
HEADER_MATCH   = re.compile(r"^(#{1,6}) (.+)$", re.MULTILINE)

# KALİTE İÇİN: Bağlam içeren, yapılandırılmış prompt
FORMULA_SYSTEM = """You are an expert at explaining mathematical formulas in research papers.

    For each formula found in the text, provide:
    1. **Formula**: (the formula itself)
    2. **Meaning**: What this formula computes or represents in 1-2 sentences
    3. **Symbols**: Each symbol and its meaning (e.g. α = learning rate)
    4. **Role in paper**: Why this formula matters in the context of this section

    Be precise. Use the surrounding text as context for better explanations.
    If formulas are related (e.g. a loss function and its gradient), explicitly connect them."""

class FormulaChunk(TypedDict):
    section: str
    content: str          # Bağlam için tam section içeriği
    formulas: List[str]


def parse_formula_chunks(md_path: str) -> List[FormulaChunk]:
    text = Path(md_path).read_text(encoding="utf-8")
    sections = HEADER_PATTERN.split(text)
    chunks: List[FormulaChunk] = []

    for section in sections:
        if not section.strip():
            continue
        header_match = HEADER_MATCH.match(section)
        section_title = header_match.group(2) if header_match else "Introduction"
        formulas = INLINE_FORMULA.findall(section) + BLOCK_FORMULA.findall(section)
        if formulas:
            chunks.append(FormulaChunk(
                section=section_title,
                content=_extract_context(section),  # ← Tam metin değil, akıllı özet
                formulas=formulas,
            ))
    return chunks


def _extract_context(section_text: str, max_chars: int = 1_500) -> str:
    """
    Section'ın tamamını değil, formül içeren cümleleri + 
    1 öncesi/sonrası cümleyi alır.
    Token tasarrufu sağlarken bağlamı korur.
    """
    sentences = re.split(r'(?<=[.!?])\s+', section_text)
    relevant = []

    for i, sent in enumerate(sentences):
        has_formula = bool(INLINE_FORMULA.search(sent) or BLOCK_FORMULA.search(sent))
        if has_formula:
            # 1 önceki cümle (bağlam)
            if i > 0 and sentences[i-1] not in relevant:
                relevant.append(sentences[i-1])
            relevant.append(sent)
            # 1 sonraki cümle (bağlam)
            if i < len(sentences) - 1:
                relevant.append(sentences[i+1])

    context = " ".join(relevant)
    return context[:max_chars]  # Hard cap


def _chunks_to_batches(chunks: List[FormulaChunk]) -> List[str]:
    """
    Her batch'e section başlığı + bağlam + formüller girer.
    LLM bağlamı görerek açıklama yapar.
    """
    batches, current = [], ""

    for c in chunks:
        formula_lines = "\n".join(f"  {f}" for f in c["formulas"])
        block = (
            f"## Section: {c['section']}\n"
            f"### Context:\n{c['content']}\n"
            f"### Formulas to explain:\n{formula_lines}"
        )
        if current and len(current) + len(block) > BATCH_CHARS:
            batches.append(current.strip())
            current = block
        else:
            current += "\n\n" + block

    if current.strip():
        batches.append(current.strip())

    return batches


def explain_all_formulas(md_path: str) -> str:
    """
    Tüm formülleri bağlamıyla birlikte açıklar.
    20-50 formül için genellikle 2-4 LLM çağrısı yeterli olur.
    """
    chunks = parse_formula_chunks(md_path)
    if not chunks:
        return "No formulas found in this file."

    batches = _chunks_to_batches(chunks)
    results = []

    for i, batch in enumerate(batches, 1):
        print(f"  [Formulas] Processing batch {i}/{len(batches)}...")
        response = model.invoke([
            {"role": "system", "content": FORMULA_SYSTEM},
            {"role": "user",   "content": batch},
        ])
        results.append(response.content)

    # Başlık ekleyerek birleştir
    header = f"# Formula Explanations\n*{len(chunks)} sections, {len(batches)} batches processed*\n\n"
    return header + "\n\n---\n\n".join(results)


def get_formula_batch(md_path: str, batch_index: int = 0) -> Dict[str, Any]:
    """Sayfalama gerekirse kullanılır (opsiyonel)."""
    chunks  = parse_formula_chunks(md_path)
    if not chunks:
        return {"batch_index": 0, "total_batches": 1,
                "has_more": False, "explanation": "No formulas found."}

    batches = _chunks_to_batches(chunks)
    total   = len(batches)
    idx     = max(0, min(batch_index, total - 1))

    response = model.invoke([
        {"role": "system", "content": FORMULA_SYSTEM},
        {"role": "user",   "content": batches[idx]},
    ])
    return {"batch_index": idx, "total_batches": total,
            "has_more": idx < total - 1, "explanation": response.content}



def build_formula_list(md_path: str) -> List[Dict]:
    """
    Makaleyi parse edip numaralı formül listesi döndürür.
    State'e kaydedilmek üzere hazırlanır.
    """
    chunks = parse_formula_chunks(md_path)
    formula_list = []
    counter = 1

    for chunk in chunks:
        for formula in chunk["formulas"]:
            formula_list.append({
                "id": counter,
                "section": chunk["section"],
                "formula": formula,
                "context": _extract_context(chunk["content"], max_chars=500),
            })
            counter += 1

    return formula_list


def explain_selected_formulas(
    formula_list: List[Dict],
    ids: List[int],
) -> str:
    """Seçili ID'lere sahip formülleri bağlamıyla açıklar."""
    selected = [f for f in formula_list if f["id"] in ids]
    if not selected:
        return "Seçilen formüller bulunamadı."

    content = "\n\n".join(
        f"## [{f['id']}] Section: {f['section']}\n"
        f"### Context:\n{f['context']}\n"
        f"### Formula:\n{f['formula']}"
        for f in selected
    )

    response = model.invoke([
        {"role": "system", "content": FORMULA_SYSTEM},
        {"role": "user",   "content": content},
    ])
    return response.content
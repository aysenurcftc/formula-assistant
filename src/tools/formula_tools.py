import re
import concurrent.futures
from functools import lru_cache
from pathlib import Path
from typing import TypedDict, List, Dict, Any
from src.config import model

BATCH_CHARS = 8000

INLINE_FORMULA = re.compile(r"(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)")
BLOCK_FORMULA  = re.compile(r"\$\$.*?\$\$", re.DOTALL)
HEADER_PATTERN = re.compile(r"(?=^#{1,6} .+$)", re.MULTILINE)
HEADER_MATCH   = re.compile(r"^(#{1,6}) (.+)$", re.MULTILINE)


FORMULA_PROMPTS = {
    "quick": """You are an expert at explaining mathematical formulas intuitively.

        For each formula:
        1. **Big Picture**: What does this formula achieve? (1 sentence)
        2. **Key Variable**: Which single variable most affects the output, and why?
        3. **Intuition**: One analogy or real-world parallel that makes it click.

        Rules:
        - No derivations, no step-by-step math.
        - Pure intuition. Dense, vivid, fast.
        - Use the surrounding context to ground your explanation.""",

            "deep": """You are an expert at explaining mathematical formulas in research papers.

        For each formula:
        1. **Formula**: State it clearly.
        2. **Meaning**: What this formula computes or represents (1-2 sentences).
        3. **Symbols**: Each symbol and its meaning (e.g. α = learning rate).
        4. **Derivation Logic**: Why does each operation exist? (e.g. why log, why square root?)
        5. **Role in paper**: Why this formula matters in this section's context.
        6. **Connections**: If related to another formula in the section, explicitly connect them.

        Be rigorous. Use the surrounding text as context.""",
}

SECTION_SUMMARY_PROMPT = """Summarize this section in max 80 words.
Focus on: main claim, method used, key result (if any).
Be dense. No filler words."""

REDUCE_PROMPT = """You are given section-level summaries of a research paper.
Write a cohesive 250-300 word summary covering:
1. Research goal & motivation
2. Proposed method / approach
3. Key results & findings
4. Conclusions & impact

Source: the section summaries below. Be precise and avoid repetition."""


#TypedDict
class FormulaChunk(TypedDict):
    section: str
    content: str
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
                content=_extract_context(section),
                formulas=formulas,
            ))
    return chunks


def _extract_context(section_text: str, max_chars: int = 1_500) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', section_text)
    relevant = []

    for i, sent in enumerate(sentences):
        has_formula = bool(INLINE_FORMULA.search(sent) or BLOCK_FORMULA.search(sent))
        if has_formula:
            if i > 0 and sentences[i - 1] not in relevant:
                relevant.append(sentences[i - 1])
            relevant.append(sent)
            if i < len(sentences) - 1:
                relevant.append(sentences[i + 1])

    return " ".join(relevant)[:max_chars]


def _chunks_to_batches(chunks: List[FormulaChunk]) -> List[str]:
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



@lru_cache(maxsize=8)
def build_formula_list(md_path: str) -> List[Dict]:
    """
    Parses the article and returns a numbered list of formulas.
    Cached by md_path: for the same file, the parsing step is skipped 
    entirely on the second call.
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
    mode: str = "deep",
) -> str:
    selected = [f for f in formula_list if f["id"] in ids]
    if not selected:
        return "The selected formulas could not be found."

    content = "\n\n".join(
        f"## [{f['id']}] Section: {f['section']}\n"
        f"### Context:\n{f['context']}\n"
        f"### Formula:\n{f['formula']}"
        for f in selected
    )

    response = model.invoke([
        {"role": "system", "content": FORMULA_PROMPTS.get(mode, FORMULA_PROMPTS["deep"])},
        {"role": "user",   "content": content},
    ])
    return response.content


def explain_all_formulas(md_path: str, mode: str = "deep") -> str:
    chunks = parse_formula_chunks(md_path)
    if not chunks:
        return "No formulas found in this file."

    batches = _chunks_to_batches(chunks)
    system_prompt = FORMULA_PROMPTS.get(mode, FORMULA_PROMPTS["deep"])
    results = []

    for i, batch in enumerate(batches, 1):
        print(f"  [Formulas] Batch {i}/{len(batches)} (mode={mode})...")
        response = model.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": batch},
        ])
        results.append(response.content)

    header = f"# Formula Explanations ({mode} mode)\n*{len(chunks)} sections, {len(batches)} batches*\n\n"
    return header + "\n\n---\n\n".join(results)



def _summarize_section(section_text: str, section_title: str) -> str:
    response = model.invoke([
        {"role": "system", "content": SECTION_SUMMARY_PROMPT},
        {"role": "user",   "content": f"## {section_title}\n\n{section_text[:2000]}"},
    ])
    return f"### {section_title}\n{response.content}"


def summarize_paper_mapreduce(md_path: str) -> str:
    text = Path(md_path).read_text(encoding="utf-8")

    raw_sections = HEADER_PATTERN.split(text)
    sections = []
    for sec in raw_sections:
        if not sec.strip():
            continue
        header = HEADER_MATCH.match(sec)
        title = header.group(2) if header else "Introduction"
        if len(sec.strip()) > 200:
            sections.append((title, sec))

    if not sections:
        return "Could not parse sections from this paper."

    print(f"  [Summary] Map: {len(sections)} sections...")

    mini_summaries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_summarize_section, sec_text, title): title
            for title, sec_text in sections
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                mini_summaries.append(future.result())
            except Exception as e:
                print(f"  [Warning] Section failed: {e}")

    print(f"  [Summary] Reduce: {len(mini_summaries)} summaries...")

    combined = "\n\n".join(mini_summaries)
    response = model.invoke([
        {"role": "system", "content": REDUCE_PROMPT},
        {"role": "user",   "content": combined},
    ])
    return response.content



def get_formula_batch(md_path: str, batch_index: int = 0, mode: str = "deep") -> Dict[str, Any]:
    chunks = parse_formula_chunks(md_path)
    if not chunks:
        return {"batch_index": 0, "total_batches": 1,
                "has_more": False, "explanation": "No formulas found."}

    batches = _chunks_to_batches(chunks)
    total   = len(batches)
    idx     = max(0, min(batch_index, total - 1))

    response = model.invoke([
        {"role": "system", "content": FORMULA_PROMPTS.get(mode, FORMULA_PROMPTS["deep"])},
        {"role": "user",   "content": batches[idx]},
    ])
    return {"batch_index": idx, "total_batches": total,
            "has_more": idx < total - 1, "explanation": response.content}
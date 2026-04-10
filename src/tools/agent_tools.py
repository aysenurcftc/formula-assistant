from langchain.tools import tool
from src.tools.pdf_tools import convert_pdf_to_md
from src.tools.formula_tools import (
    build_formula_list,
    explain_all_formulas,
    explain_selected_formulas,
    summarize_paper_mapreduce,
)
from typing import List


@tool
def convert_pdf(pdf_path: str) -> str:
    """Converts a PDF to markdown. Must be called first."""
    return convert_pdf_to_md(pdf_path)


@tool
def list_formulas(md_path: str) -> str:
    """
    Parses the paper and returns a numbered list of all formulas.
    Call this FIRST when user wants to see or select formulas.
    Returns: numbered list with section info.
    """
    formula_list = build_formula_list(md_path)
    print(f"\n  [TOOL CALLED] list_formulas → {len(formula_list)} formulas found")
    if not formula_list:
        return "No formulas found."

    lines = [f"Found {len(formula_list)} formulas:\n"]
    for f in formula_list:
        lines.append(f"[{f['id']}] ({f['section']}) {f['formula']}")

    return "\n".join(lines)


@tool
def explain_formulas_by_ids(
    md_path: str,
    formula_ids: List[int],
    mode: str = "deep",
) -> str:
    """
    Explains specific formulas by their ID numbers.
    Use when user selects one or more formulas from the list.

    Args:
        md_path: Path to the markdown file.
        formula_ids: List of integer IDs shown in list_formulas output.
        mode: Explanation style.
            'quick' → intuitive big-picture (key variable + one analogy, no math).
            'deep'  → step-by-step derivation with symbol table (default).
    """
    formula_list = build_formula_list(md_path)
    result = explain_selected_formulas(formula_list, formula_ids, mode=mode)
    # Prefix lets supervisor_node track which IDs were explained this turn.
    return f"EXPLAINED_IDS:{formula_ids}\n\n{result}"


@tool
def explain_all_formulas_tool(md_path: str, mode: str = "deep") -> str:
    """
    Explains ALL formulas in the paper at once.
    Use when user wants everything explained without selection.

    Args:
        md_path: Path to the markdown file.
        mode: 'quick' or 'deep' (default: 'deep').
    """
    return explain_all_formulas(md_path, mode=mode)


@tool
def summarize_paper(md_path: str) -> str:
    """
    Summarizes the paper using map-reduce over all sections.
    Each section is summarized in parallel, then merged into a
    cohesive 250-300 word final summary.
    """
    return summarize_paper_mapreduce(md_path)
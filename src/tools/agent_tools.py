from langchain.tools import tool
from src.tools.pdf_tools import convert_pdf_to_md
from src.config import model
from src.tools.formula_tools import (
    build_formula_list,
    explain_all_formulas,
    explain_selected_formulas,
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
def explain_formulas_by_ids(md_path: str, formula_ids: List[int]) -> str:
    """
    Explains specific formulas by their ID numbers.
    Use when user selects one or more formulas from the list.
    formula_ids: list of integer IDs shown in list_formulas output.
    """
    formula_list = build_formula_list(md_path)
    return explain_selected_formulas(formula_list, formula_ids)


@tool
def explain_all_formulas_tool(md_path: str) -> str:
    """
    Explains ALL formulas in the paper at once.
    Use when user wants everything explained without selection.
    """
    return explain_all_formulas(md_path)


@tool
def summarize_paper(md_path: str) -> str:
    """Summarizes the paper. Max 300 words."""
    with open(md_path, encoding="utf-8") as f:
        content = f.read()[:5000]

    response = model.invoke([
        {
            "role": "system",
            "content": (
                "Summarize in max 300 words: "
                "main goal, methods, key findings, conclusions."
            ),
        },
        {"role": "user", "content": content},
    ])
    return response.content
import ast
import re
from typing import List
from functools import lru_cache
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.postgres import PostgresSaver

from src.config import model
from src.tools.agent_tools import (
    convert_pdf,
    list_formulas,
    explain_formulas_by_ids,
    explain_all_formulas_tool,
    summarize_paper,
)
from src.memory.memory import (
    MemoryState, build_context,
    summarize_and_trim, needs_summarization,
)


@lru_cache(maxsize=1)
def _get_supervisor():
    return create_react_agent(
        model,
        tools=[
            convert_pdf,
            list_formulas,
            explain_formulas_by_ids,
            explain_all_formulas_tool,
            summarize_paper,
        ],
        prompt="""You are a research paper assistant. Always respond in the user's language.

            STRICT RULES — NEVER BREAK:
            - ALWAYS call list_formulas before listing or explaining formulas.
            - NEVER read the markdown file directly to extract formulas.
            - NEVER generate formula lists from memory.
            - Formula IDs come ONLY from list_formulas tool output.

            WORKFLOW:
            1. Always call convert_pdf first → get md_path.

            2. Based on user intent:
            - "list formulas" / "show formulas"    → list_formulas(md_path)
            - "explain formula [N]"                → explain_formulas_by_ids(md_path, [N], mode)
            - "explain remaining" / "explain rest" → explain_formulas_by_ids(md_path, <remaining_ids>, mode)
            - "explain all formulas"               → explain_all_formulas_tool(md_path, mode)
            - "summarize" / paper questions        → summarize_paper(md_path)

            3. Mode selection (pass as `mode` argument):
            - User says "quick", "hızlı", "sezgisel", "big picture" → mode='quick'
            - User says "deep", "detaylı", "adım adım", "türet"     → mode='deep'
            - Default (no preference stated)                         → mode='deep'

            4. For "explain remaining":
            - The system context above lists already-explained IDs.
            - Pass only the IDs NOT in that list to explain_formulas_by_ids.

            5. After listing formulas always ask:
            "Which formula(s) would you like explained? (Enter numbers, 'all', or 'remaining')"
            "Quick intuition or deep derivation?"
            """,
    )


_EXPLAINED_RE = re.compile(r"EXPLAINED_IDS:(\[[^\]]*\])")


def _extract_explained_ids(messages, state: MemoryState) -> List[int]:
    existing = set(state.get("explained_ids") or [])
    for msg in messages:
        if getattr(msg, "type", None) == "tool":
            content = getattr(msg, "content", "") or ""
            match = _EXPLAINED_RE.search(content)
            if match:
                try:
                    ids = ast.literal_eval(match.group(1))
                    existing.update(ids)
                except (ValueError, SyntaxError):
                    pass
    return list(existing)


#supervisor node 
def supervisor_node(state: MemoryState):
    context = build_context(state)

    formula_list  = state.get("formula_list") or []
    explained_ids = state.get("explained_ids") or []

    if formula_list:
        remaining  = [f["id"] for f in formula_list if f["id"] not in explained_ids]
        cache_info = (
            f"Formula cache active: {len(formula_list)} formulas loaded. "
            f"Explained IDs: {explained_ids}. "
            f"Remaining IDs: {remaining}."
        )
        context = [SystemMessage(content=cache_info)] + context

    result = _get_supervisor().invoke({"messages": context})

    new_explained = _extract_explained_ids(result["messages"], state)
    updates = {"messages": result["messages"]}
    if new_explained:
        updates["explained_ids"] = new_explained

    return updates


#graph builder

def build_graph(checkpointer: PostgresSaver) -> StateGraph:
    builder = StateGraph(MemoryState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("summarize",  summarize_and_trim)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor", needs_summarization,
        {"summarize": "summarize", "__end__": "__end__"},
    )
    builder.add_edge("summarize", "__end__")
    return builder.compile(checkpointer=checkpointer)


def make_checkpointer(db_uri: str) -> PostgresSaver:
    """
    It opens the psycopg connection directly and passes it to PostgresSaver.
    It does not use a context manager—this allows the connection to remain open 
    throughout the process via Streamlit's cache_resource.
    """
    import psycopg
    conn = psycopg.connect(db_uri, autocommit=True)
    return PostgresSaver(conn)
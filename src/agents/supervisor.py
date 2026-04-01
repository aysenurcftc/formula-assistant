from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.postgres import PostgresSaver
from functools import lru_cache
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

            STRICT RULES - NEVER BREAK:
            - ALWAYS call list_formulas tool before listing or explaining formulas.
            - NEVER read the markdown file directly to extract formulas.
            - NEVER generate formula lists from memory.
            - Formula IDs come ONLY from list_formulas tool output.

            WORKFLOW:
            1. Always call convert_pdf first → get md_path.

            2. Based on user intent:
            - "list formulas" / "show formulas"   → list_formulas(md_path)
            - "explain formula [N]" or user picks → explain_formulas_by_ids(md_path, [N])
            - "explain remaining" / "explain rest"→ explain_formulas_by_ids(md_path, <unlisted ids>)
            - "explain all formulas"              → explain_all_formulas_tool(md_path)
            - "summarize" / paper questions       → summarize_paper(md_path)

            3. For "explain remaining":
            - Check which IDs were already explained in conversation history.
            - Pass only the unexplained IDs to explain_formulas_by_ids.

            4. After listing formulas, always ask: 
            "Which formula(s) would you like explained? (Enter numbers, or say 'all' / 'remaining')"
            
            """,
    )


def supervisor_node(state: MemoryState):
    context = build_context(state)
    result  = _get_supervisor().invoke({"messages": context})
    return {"messages": result["messages"]}


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


def get_graph(db_uri: str):
    return PostgresSaver.from_conn_string(db_uri)




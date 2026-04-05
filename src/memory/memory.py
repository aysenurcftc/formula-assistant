from src.config import model
from typing import List, Dict, Any
from langchain_core.messages import (
    HumanMessage, SystemMessage, RemoveMessage,
    AIMessage, ToolMessage,
)
from langgraph.graph.message import MessagesState
from src.config import TRIM_THRESHOLD, KEEP_RECENT, MAX_TOKENS


class MemoryState(MessagesState):
    summary: str
    formula_list: List[Dict]
    explained_ids: List[int]
    current_md_path: str


def count_tokens(messages: List[Any]) -> int:
    total_words = sum(
        len(str(getattr(m, "content", m)).split())
        for m in messages
    )
    return int(total_words * 1.33)


def _safe_trim(messages: List[Any], max_tokens: int) -> List[Any]:
    """
    Manual trimming instead of trim_messages:
    Appends messages from end to beginning.
    Never leaves orphaned tool_use or tool_result messages.
    tool_use + tool_result pairs always stay together or are removed together.
    """
    # First, identify tool_use/tool_result pairs.
    # If an AIMessage contains a tool_use block, it must be handled together with the immediately following ToolMessage.

    kept = []
    total = 0
    i = len(messages) - 1

    while i >= 0:
        msg = messages[i]
        if isinstance(msg, ToolMessage):
            if i > 0 and isinstance(messages[i - 1], AIMessage):
                pair = [messages[i - 1], msg]
                cost = count_tokens(pair)
                if total + cost <= max_tokens:
                    kept = pair + kept
                    total += cost
                i -= 2
            else:
                i -= 1
            continue

        content = getattr(msg, "content", None)
        has_tool_use = (
            isinstance(content, list) and
            any(
                isinstance(b, dict) and b.get("type") == "tool_use"
                for b in content
            )
        )
        if has_tool_use:
            if i + 1 < len(messages) and isinstance(messages[i + 1], ToolMessage):
                pair = [msg, messages[i + 1]]
                cost = count_tokens(pair)
                if total + cost <= max_tokens:
                    if messages[i + 1] not in kept:
                        kept = pair + kept
                        total += cost
            else:
                cost = count_tokens([msg])
                if total + cost <= max_tokens:
                    kept = [msg] + kept
                    total += cost
            i -= 1
            continue

        cost = count_tokens([msg])
        if total + cost <= max_tokens:
            kept = [msg] + kept
            total += cost
        i -= 1

    while kept and not isinstance(kept[0], HumanMessage):
        kept = kept[1:]

    return kept


def build_context(state: MemoryState) -> List:
    summary  = state.get("summary", "")
    messages = list(state["messages"])

    summary_msg = []
    if summary:
        summary_msg = [SystemMessage(content=f"Conversation Summary:\n{summary}")]

    trimmed = _safe_trim(messages, MAX_TOKENS)
    return summary_msg + trimmed


def summarize_and_trim(state: MemoryState) -> Dict[str, Any]:
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    messages_to_summarize = messages[:-KEEP_RECENT]

    if existing_summary:
        prompt = (
            f"Current summary:\n{existing_summary}\n\n"
            "Update the summary to include the new messages above. "
            "Maintain info about which PDFs were analyzed, the questions asked, "
            "and key answers given. Max 200 words."
        )
    else:
        prompt = (
            "Summarize this conversation. Note which PDFs were analyzed, "
            "the questions asked, and the key answers provided. Max 200 words."
        )

    response = model.invoke(messages_to_summarize + [HumanMessage(content=prompt)])
    new_summary = response.content

    to_delete = [RemoveMessage(id=m.id) for m in messages_to_summarize]

    print(
        f"\n  [Memory] {len(messages)} messages — summarized, "
        f"{len(to_delete)} old messages cleared.\n"
    )

    return {"summary": new_summary, "messages": to_delete}


def needs_summarization(state: MemoryState) -> str:
    if len(state["messages"]) >= TRIM_THRESHOLD:
        return "summarize"
    return "__end__"
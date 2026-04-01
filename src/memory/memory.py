
from src.config import model
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage, trim_messages
from langgraph.graph.message import MessagesState
from src.config import TRIM_THRESHOLD, KEEP_RECENT, MAX_TOKENS


class MemoryState(MessagesState):
    summary: str
    formula_list: List[Dict]   # [{"id": 1, "section": "...", "formula": "..."}]
    explained_ids: List[int]   # Açıklananların id listesi
    current_md_path: str       # Aktif markdown dosyası


def count_tokens(messages: List[Any]) -> int:
    """Approximate token counter (Words * 1.33)."""
    total_words = sum(
        len(str(getattr(m, "content", m)).split())
        for m in messages
    )
    return int(total_words * 1.33)

def build_context(state: MemoryState) -> List:
    """Builds the context by combining summary and trimmed messages."""
    summary = state.get("summary", "")
    messages = list(state["messages"])

    summary_msg = []
    if summary:
        summary_msg = [SystemMessage(content=f"Conversation Summary:\n{summary}")]

    # Trim conversation messages
    trimmed = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens,
        max_tokens=MAX_TOKENS,
        start_on="human",
        end_on=("human", "tool"),
        include_system=False, # Don't trim existing system messages
    )

    return summary_msg + trimmed

def summarize_and_trim(state: MemoryState) -> Dict[str, Any]:
    """Summarizes old messages and returns RemoveMessage commands."""
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    # We summarize everything except the most recent ones (KEEP_RECENT)
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

    # Invoke model with the part to be deleted + the summary prompt
    response = model.invoke(messages_to_summarize + [HumanMessage(content=prompt)])
    new_summary = response.content

    # Create RemoveMessage objects for old history
    to_delete = [RemoveMessage(id=m.id) for m in messages_to_summarize]

    print(
        f"\n  [Memory] {len(messages)} messages accumulated — summarized, "
        f"{len(to_delete)} old messages cleared.\n"
    )

    return {"summary": new_summary, "messages": to_delete}

def needs_summarization(state: MemoryState) -> str:
    """Conditional edge logic to check if memory cleanup is needed."""
    if len(state["messages"]) >= TRIM_THRESHOLD:
        return "summarize"
    return "__end__"
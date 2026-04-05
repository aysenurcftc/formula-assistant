import uuid
import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from src.config import DB_URI
from main import ensure_postgres


st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Research Paper Assistant")
st.caption("Upload a PDF and ask questions about its content and formulas.")



def _init_state():
    defaults = {
        "thread_id":    None,
        "messages":     [],
        "pdf_path":     None,
        "pdf_name":     None,
        "api_key_used": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


#Sidebar 

with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-…",
        help="Used only in this session, never stored.",
    )

    st.divider()
    st.header("Upload Paper")

    uploaded = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")

    if uploaded:
        if uploaded.name != st.session_state.pdf_name:
            if st.session_state.pdf_path:
                Path(st.session_state.pdf_path).unlink(missing_ok=True)

            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf", dir=tempfile.gettempdir()
            )
            tmp.write(uploaded.read())
            tmp.flush()

            st.session_state.pdf_path  = tmp.name
            st.session_state.pdf_name  = uploaded.name
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages  = []
            st.success(f"✓ {uploaded.name}")

    if st.session_state.pdf_name:
        st.divider()
        st.markdown(f"**Active:** {st.session_state.pdf_name}")
        st.caption(f"Thread: `{st.session_state.thread_id[:8]}…`")
        if st.button("🗑 New session", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages  = []
            st.rerun()

    st.divider()
    st.markdown("**Tip — formula modes**")
    st.caption("Add **(quick)** or **(deep)** to your message.")
    st.caption("Example: *Explain formula 3 (quick)*")



if not api_key:
    st.info("Enter your Anthropic API key in the sidebar to get started.")
    st.stop()

if not api_key.startswith("sk-ant-"):
    st.warning("That doesn't look like a valid Anthropic API key.")
    st.stop()


#DB + graph 

@st.cache_resource(show_spinner="Connecting…")
def _get_graph(key: str):
    """
    The api_key is passed as a parameter: a key change results in a cache miss, causing it to be regenerated.
    make_checkpointer does not use a context manager; the connection remains open for the duration of the process.

    """
    import os
    os.environ["ANTHROPIC_API_KEY"] = key

    from src.agents.supervisor import build_graph, make_checkpointer

    ensure_postgres(DB_URI)

    checkpointer = make_checkpointer(DB_URI)
    try:
        checkpointer.setup()
    except Exception:
        pass

    return build_graph(checkpointer)


try:
    graph = _get_graph(api_key)
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()


if not st.session_state.pdf_path:
    st.info("Upload a PDF from the sidebar to continue.")
    st.stop()


for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Ask about the paper…")

if user_input:
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    message = f"PDF: {st.session_state.pdf_path}\n\nUser request: {user_input}"
    config  = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.spinner("Thinking…"):
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=message)]},
                config,
            )
            reply = ""
            for msg in reversed(result["messages"]):
                if getattr(msg, "type", None) == "ai" and msg.content:
                    reply = msg.content
                    break
            if not reply:
                reply = "_(no response)_"
        except Exception as e:
            reply = f"**Error:** {e}"

    st.session_state.messages.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)
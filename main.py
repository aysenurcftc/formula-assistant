
import sys
import uuid
import signal
import argparse
import time
import subprocess
import psycopg2
from pathlib import Path


from langchain_core.messages import HumanMessage
from src.config import DB_URI
from src.agents.supervisor import build_graph, get_graph

def ensure_postgres(db_uri: str = DB_URI) -> None:
    """Checks if Postgres is running in Docker, starts it if not."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", "langgraph-postgres"],
        capture_output=True, text=True
    )
    if result.stdout.strip() != "true":
        print("  [DB] Starting PostgreSQL...")
        subprocess.run(["docker", "compose", "up", "-d"], check=True)

    _wait_for_postgres(db_uri)

def _wait_for_postgres(db_uri: str, retries: int = 15, delay: float = 1.5) -> None:
    """Waits for Postgres to become ready for connections."""
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(db_uri)
            conn.close()
            print("  [DB] PostgreSQL is ready.\n")
            return
        except psycopg2.OperationalError:
            print(f"  [DB] Waiting... ({attempt}/{retries})")
            time.sleep(delay)
    raise RuntimeError(
        "Could not start PostgreSQL. "
        "Check logs with `docker compose logs postgres`."
    )

def run_chat(pdf_path: str, db_uri: str) -> None:
    ensure_postgres(db_uri)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 60)
    print("  Research Paper Assistant")
    print("=" * 60)
    print(f"  PDF      : {pdf_path}")
    print(f"  Thread ID: {thread_id}")
    print("  Type 'exit' to quit or press Ctrl+C.")
    print("=" * 60 + "\n")

    with get_graph(db_uri) as checkpointer:
        try:
            checkpointer.setup()
        except Exception:
            pass  # Skip if tables already exist

        graph = build_graph(checkpointer)

        def cleanup():
            try:
                # Optional: Delete thread from DB on exit to keep it clean
                # checkpointer.delete_thread(thread_id) 
                print("\n  [Memory] Session ended.")
            except Exception as e:
                print(f"\n  [Warning] Could not clean session: {e}")

        def handle_signal(sig, frame):
            cleanup()
            print("  Goodbye!\n")
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_signal)

        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                break

            message = f"PDF: {pdf_path}\n\nUser request: {user_input}"

            print("\nAssistant: ", end="", flush=True)
            try:
                result = graph.invoke(
                    {"messages": [HumanMessage(content=message)]},
                    config,
                )
                # Get the last AI message from the result
                for msg in reversed(result["messages"]):
                    if msg.type == "ai" and msg.content:
                        print(msg.content)
                        break
            except Exception as e:
                print(f"[Error] {e}")

            print()

        cleanup()
        print("  Goodbye!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Paper Assistant CLI")
    parser.add_argument("pdf", help="Path to the PDF file to analyze")
    parser.add_argument("--db", default=DB_URI, help="PostgreSQL connection URI")
    args = parser.parse_args()

    pdf_file = Path(args.pdf)
    if not pdf_file.exists():
        print(f"[Error] PDF not found: {pdf_file}")
        sys.exit(1)

    run_chat(str(pdf_file), args.db)
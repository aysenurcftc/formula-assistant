# formula-assistant

A LangGraph-powered agent for exploring research papers — built around formula explanation.

Upload a PDF, list its formulas, and get either a quick intuition or a deep step-by-step derivation for any of them.

## Features

- **Formula listing** — extracts and numbers every LaTeX formula with its section context
- **Two explanation modes**
  - `quick` — big picture, key variable, one analogy. No math.
  - `deep` — symbol table, derivation logic, connections between formulas
- **Map-reduce summary** — sections summarized in parallel, merged into a 250-300 word overview
- **Persistent memory** — history stored in PostgreSQL; old messages auto-summarized when context grows
- **Streamlit UI** — upload PDF, enter API key, start chatting

## How It Works
1. **Marker** converts the PDF to Markdown, preserving LaTeX formulas.
2. The agent calls `list_formulas` to parse and number every `$...$` and `$$...$$` block.
3. Formula list is cached in-process (`lru_cache`) — subsequent calls skip re-parsing.
4. The user picks formulas by ID; the agent calls `explain_formulas_by_ids` with the chosen mode.
5. `explained_ids` are tracked in LangGraph state so "explain remaining" always works correctly.

## Stack

| Layer | Library |
|---|---|
| PDF → Markdown | `marker-pdf` |
| Agent / memory | `langgraph`, `langchain-anthropic` |
| Checkpointing | `PostgresSaver` (`psycopg`) |
| UI | `streamlit` |
| Package manager | `uv` |

## Setup

**1. Start PostgreSQL**
```bash
docker compose up -d
```

**2. Install dependencies**
```bash
uv sync
```

**3. Configure environment** — optional, key can also be entered in the UI


**4. Run**
```bash
# Streamlit UI
uv run streamlit run app.py

# CLI
uv run python main.py path/to/paper.pdf
```

## Usage

| Intent | What to type |
|---|---|
| List all formulas | `list the formulas` |
| Quick intuition | `explain formula 3 (quick)` |
| Deep derivation | `explain formula 3 (deep)` |
| Explain remaining | `explain remaining (quick)` |
| Summarize paper | `summarize` |

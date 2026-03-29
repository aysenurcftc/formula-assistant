from pathlib import Path
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent 
from src.tools.pdf_tools import convert_pdf_to_md      
from src.tools.formula_tools import parse_formula_chunks

load_dotenv()

model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
)


@tool
def convert_pdf(pdf_path: str) -> str:
    """Converts a PDF file to markdown. Should always be called first."""
    return convert_pdf_to_md(pdf_path)


@tool
def extract_and_explain_formulas(md_path: str) -> str:
    """Extracts and explains formulas found in the markdown file."""
    chunks = parse_formula_chunks(md_path)
    if not chunks:
        return "No formulas found in this file."
    results = []
    for chunk in chunks:
        formula_list = "\n".join(f"  - {f}" for f in chunk["formulas"])
        results.append(f"### {chunk['section']}\nFormulas:\n{formula_list}\nContent:\n{chunk['content']}")
    return "\n\n".join(results)


@tool
def read_markdown(md_path: str) -> str:
    """Reads the content of a markdown file. Used for summarization."""
    with open(md_path, encoding="utf-8") as f:
        return f.read()[:4000]


formula_agent = create_react_agent(
    model,
    tools=[extract_and_explain_formulas],
    prompt=(
        "You are a math formula expert. "
        "Extract formulas from the markdown file and explain each one clearly: "
        "what it represents, what each symbol means, and its mathematical significance. "
        "Be concise but complete."
    )
)

summary_agent = create_react_agent(
    model,
    tools=[read_markdown],
    prompt=(
        "You are a scientific paper summarizer. "
        "Read the markdown file and summarize the paper in max 300 words: "
        "main objective, methods, key findings, conclusions. "
        "Plain English."
    )
)


@tool
def explain_formulas(request: str) -> str:
    """Explains the mathematical formulas in the paper.
    Use when the user wants to understand formulas, learn about symbols,
    or get explanations for mathematical expressions.
    Input: markdown file path and what the user wants
    """
    result = formula_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content


@tool
def summarize_paper(request: str) -> str:
    """Summarizes the paper.
    Use when the user wants to know what the paper is about,
    its main findings, or its general content.
    Input: markdown file path and what the user wants
    """
    result = summary_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content


supervisor = create_react_agent(
    model,
    tools=[convert_pdf, explain_formulas, summarize_paper],
    prompt=(
        "You are a research paper assistant. "
        "When given a PDF path and a user request: "
        "1. ALWAYS call convert_pdf first to get the markdown path. "
        "2. Then call explain_formulas or summarize_paper based on the request. "
        "Pass the markdown path in your tool calls. "
        "Respond in the same language as the user."
    )
)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--prompt", "-p", required=True)
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        sys.exit(1)

    user_message = f"PDF: {pdf_path}\n\nUser request: {args.prompt}"
    print(f"\nRequest: {args.prompt}\n" + "=" * 60)

    for step in supervisor.stream(
        {"messages": [{"role": "user", "content": user_message}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()
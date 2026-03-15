"""
HR Assistant — Employee Handbooks & Internal Policies
======================================================
Interactive CLI for HR documents, oriented towards employees.
Uses the "hr" system prompt from src/prompts.py.
Friendly tone. Each answer ends with a reminder to contact HR directly.

Usage:
    python use_cases/hr/main_hr.py <pdf_path>
    python use_cases/hr/main_hr.py <pdf_path> --model claude-sonnet-4-6
"""

import argparse
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.pdf_extractor import extract_text, get_doc_stats
from src.llm_client import LLMClient, DEFAULT_MODELS
from src.prompts import get_prompt

# ──────────────────────────────────────────────
# Domain configuration
# ──────────────────────────────────────────────

SUGGESTED_QUESTIONS = [
    "How many vacation days am I entitled to?",
    "What is the remote work policy?",
    "How do I request a leave of absence or time off?",
    "What is the performance review process?",
]

HR_FOOTER = "💡 For more details, contact HR directly."

DIVIDER  = "─" * 66
DIVIDER2 = "═" * 66
WRAP_W   = 62

# ──────────────────────────────────────────────
# Welcome screen
# ──────────────────────────────────────────────

def print_header(pdf_name: str, stats: dict, model: str):
    print(f"\n{DIVIDER2}")
    print("  HR Assistant — Employee Handbook & Policies")
    print(DIVIDER)
    print(f"  Document : {pdf_name}")
    print(f"  Words    : {stats['words']:,}  |  Tokens: ~{stats['estimated_tokens']:,}")
    ctx_status = "fits in context" if stats["safe_for_context"] else "large document"
    print(f"  Context  : {ctx_status}")
    print(f"  Model    : {model}")
    print(DIVIDER2)


def print_suggestions():
    print("\n  Suggested questions:\n")
    for i, q in enumerate(SUGGESTED_QUESTIONS, start=1):
        wrapped = textwrap.fill(q, width=WRAP_W, subsequent_indent="       ")
        print(f"  [{i}]  {wrapped}")
    print()
    print("  Type a question number to use a suggestion,")
    print("  or type your own question directly.")
    print("  Commands: 'suggest' to show this list again | 'exit' to quit")
    print(DIVIDER)


# ──────────────────────────────────────────────
# Answer display
# ──────────────────────────────────────────────

def print_answer(answer: str, input_tokens: int, output_tokens: int,
                 cost: float, elapsed: float):
    print(f"\n{DIVIDER}")
    print("  Answer:\n")
    for line in textwrap.wrap(answer, width=WRAP_W):
        print(f"  {line}")
    print()
    print(f"  {HR_FOOTER}")
    print()
    print(
        f"  tokens: {input_tokens:,} in / {output_tokens} out  "
        f"| ${cost:.4f}  |  {elapsed:.1f}s"
    )
    print(DIVIDER)


# ──────────────────────────────────────────────
# REPL
# ──────────────────────────────────────────────

def run_repl(text: str, system_prompt: str, client: LLMClient, model: str):
    import time

    print_suggestions()
    session_cost  = 0.0
    session_calls = 0

    while True:
        try:
            raw = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Session ended. Take care!")
            break

        if not raw:
            continue

        if raw.lower() in ("exit", "quit", "q"):
            print(f"\n  Session finished. Have a great day!")
            print(f"  Queries made  : {session_calls}")
            print(f"  Total cost    : ${session_cost:.4f} USD")
            print()
            break

        if raw.lower() in ("suggest", "s", "help", "h"):
            print_suggestions()
            continue

        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(SUGGESTED_QUESTIONS):
                question = SUGGESTED_QUESTIONS[idx - 1]
                print(f"\n  -> {question}")
            else:
                print(f"  Invalid number. Choose between 1 and {len(SUGGESTED_QUESTIONS)}.")
                continue
        else:
            question = raw

        print("  ...", end="\r", flush=True)
        try:
            t0 = time.perf_counter()
            result = client.ask(
                question=question,
                context=text,
                system_prompt=system_prompt,
                model=model,
                temperature=0.1,
                max_tokens=600,
            )
            elapsed = time.perf_counter() - t0
        except Exception as e:
            print(f"  Error querying the model: {e}")
            continue

        session_cost  += result["cost_usd"]
        session_calls += 1

        print_answer(
            result["answer"],
            result["input_tokens"],
            result["output_tokens"],
            result["cost_usd"],
            elapsed,
        )


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HR assistant for employee handbooks and internal policies"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODELS["anthropic"],
        help=f"Claude model to use (default: {DEFAULT_MODELS['anthropic']})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"\n  Error: file not found -> {args.pdf_path}")
        sys.exit(1)

    print("  Loading document...", end="\r", flush=True)
    try:
        text  = extract_text(args.pdf_path)
        stats = get_doc_stats(text)
    except Exception as e:
        print(f"\n  Error reading PDF: {e}")
        sys.exit(1)

    system_prompt = get_prompt("hr")

    try:
        client = LLMClient(provider="anthropic")
    except EnvironmentError as e:
        print(f"\n  Error: {e}")
        print("  Add ANTHROPIC_API_KEY to your .env file")
        sys.exit(1)

    print_header(os.path.basename(args.pdf_path), stats, args.model)
    run_repl(text, system_prompt, client, args.model)


if __name__ == "__main__":
    main()

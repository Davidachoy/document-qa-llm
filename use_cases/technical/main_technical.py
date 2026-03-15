"""
Technical Assistant — API Docs, Manuals & Specifications
=========================================================
Interactive CLI specialized in technical documentation for developers.
Uses the "technical" system prompt from src/prompts.py.

Detects code blocks (``` ... ```) in model responses and renders them
with a boxed format so they stand out from prose text.

Usage:
    python use_cases/technical/main_technical.py <pdf_path>
    python use_cases/technical/main_technical.py <pdf_path> --model claude-sonnet-4-6
"""

import argparse
import os
import re
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
    "How does API authentication work?",
    "What endpoints are available and what do they do?",
    "What are the rate limiting rules and quotas?",
    "Are there code examples? Show me one.",
]

DIVIDER  = "─" * 70
DIVIDER2 = "═" * 70
WRAP_W   = 66
CODE_W   = 64   # inner width of the code box

# ──────────────────────────────────────────────
# Code block renderer
# ──────────────────────────────────────────────

# Matches fenced code blocks: ```[lang]\n...\n```
_CODE_FENCE = re.compile(r"```(\w*)\n?(.*?)```", re.DOTALL)


def _render_code_block(lang: str, code: str) -> str:
    """
    Render a code snippet inside a terminal box.

    Example output:
      ╭─ python ──────────────────────────────────────────────────╮
      │ import requests                                           │
      │ r = requests.get(url, headers={"Authorization": token})  │
      ╰───────────────────────────────────────────────────────────╯
    """
    label   = f" {lang} " if lang else " code "
    fill    = "─" * (CODE_W - len(label))
    top     = f"  ╭─{label}{fill}╮"
    bottom  = f"  ╰{'─' * (CODE_W + 2)}╯"

    lines = [top]
    for raw_line in code.strip().splitlines():
        # Wrap long lines to CODE_W
        if len(raw_line) <= CODE_W:
            padded = raw_line.ljust(CODE_W)
            lines.append(f"  │ {padded} │")
        else:
            for chunk in textwrap.wrap(raw_line, width=CODE_W):
                padded = chunk.ljust(CODE_W)
                lines.append(f"  │ {padded} │")
    lines.append(bottom)
    return "\n".join(lines)


def print_answer(answer: str, input_tokens: int, output_tokens: int,
                 cost: float, elapsed: float):
    """
    Print the answer, rendering any ``` code blocks as boxed sections
    and wrapping prose text normally.
    """
    print(f"\n{DIVIDER}")
    print("  Answer:\n")

    # Split the answer on code fences, alternating prose / code
    parts = _CODE_FENCE.split(answer)
    # _CODE_FENCE has 2 capture groups (lang, code), so split produces:
    #   [prose, lang, code, prose, lang, code, ...]
    i = 0
    while i < len(parts):
        if i % 3 == 0:
            # Prose segment
            prose = parts[i].strip()
            if prose:
                for line in textwrap.wrap(prose, width=WRAP_W):
                    print(f"  {line}")
                print()
        elif i % 3 == 1:
            # Language tag — consumed together with the next segment (code body)
            lang = parts[i].strip()
            code = parts[i + 1] if i + 1 < len(parts) else ""
            print(_render_code_block(lang, code))
            print()
            i += 1   # skip the code body (already consumed)
        i += 1

    print(
        f"  tokens: {input_tokens:,} in / {output_tokens} out  "
        f"| ${cost:.4f}  |  {elapsed:.1f}s"
    )
    print(DIVIDER)


# ──────────────────────────────────────────────
# Welcome screen
# ──────────────────────────────────────────────

def print_header(pdf_name: str, stats: dict, model: str):
    print(f"\n{DIVIDER2}")
    print("  Technical Assistant — API Docs, Manuals & Specs")
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
            print("\n\n  Session ended.")
            break

        if not raw:
            continue

        if raw.lower() in ("exit", "quit", "q"):
            print(f"\n  Session finished.")
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
                max_tokens=800,   # higher limit — code answers tend to be longer
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
        description="Technical assistant for API docs, manuals, and specs"
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

    system_prompt = get_prompt("technical")

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

"""
Experiment 02 — Claude vs OpenAI
=================================
Runs the same 5 questions on Claude Haiku and GPT-4o-mini over the same PDF
and produces a side-by-side comparison table per question, then a final
summary with totals and a cost/quality recommendation.

Usage:
    python experiments/02_claude_vs_openai.py <pdf_path> [--domain legal|technical|hr]
"""

import argparse
import os
import sys
import textwrap
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_extractor import extract_text, get_doc_stats
from src.llm_client import LLMClient, DEFAULT_MODELS
from src.prompts import get_prompt

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

PROVIDERS = [
    {"key": "anthropic", "label": "Claude Haiku"},
    {"key": "openai",    "label": "GPT-4o-mini"},
]

QUESTIONS = [
    "What is the duration or term of this document?",
    "Who are the parties or subjects involved in this document?",
    "What are the main obligations or responsibilities mentioned?",
    "What conditions, restrictions, or limitations are established?",
    "What is the main purpose or objective of this document?",
]

MAX_TOKENS = 300

# Table column widths
W_PROVIDER = 16
W_ANSWER   = 42
W_TOKENS   = 13
W_COST     = 9
W_TIME     = 7

# ──────────────────────────────────────────────
# Table helpers
# ──────────────────────────────────────────────

def _row(provider: str, answer: str, in_tok: int, out_tok: int, cost: float, secs: float) -> list[str]:
    """Split one result into display rows (answer may wrap across multiple lines)."""
    answer_lines = textwrap.wrap(answer, width=W_ANSWER) or ["(empty)"]
    tokens_str = f"{in_tok:,}/{out_tok}"
    cost_str   = f"${cost:.4f}"
    time_str   = f"{secs:.1f}s"

    rows = []
    for i, line in enumerate(answer_lines):
        if i == 0:
            rows.append((provider, line, tokens_str, cost_str, time_str))
        else:
            rows.append(("", line, "", "", ""))
    return rows


def _sep(left: str, mid: str, right: str, fill: str = "─") -> str:
    parts = [
        fill * (W_PROVIDER + 2),
        fill * (W_ANSWER   + 2),
        fill * (W_TOKENS   + 2),
        fill * (W_COST     + 2),
        fill * (W_TIME     + 2),
    ]
    return left + mid.join(parts) + right


def _data_row(cols: tuple) -> str:
    provider, answer, tokens, cost, secs = cols
    return (
        f"│ {provider:<{W_PROVIDER}} "
        f"│ {answer:<{W_ANSWER}} "
        f"│ {tokens:<{W_TOKENS}} "
        f"│ {cost:<{W_COST}} "
        f"│ {secs:<{W_TIME}} │"
    )


def print_question_table(q_index: int, question: str, results: list[dict]):
    wrapped_q = textwrap.fill(question, width=70, subsequent_indent="    ")
    print(f"\n{'═' * 82}")
    print(f"  Question {q_index}: {wrapped_q}")
    print(_sep("┌", "┬", "┐"))

    # Header
    header = (
        f"│ {'Provider':<{W_PROVIDER}} "
        f"│ {'Answer':<{W_ANSWER}} "
        f"│ {'Tokens in/out':<{W_TOKENS}} "
        f"│ {'Cost':<{W_COST}} "
        f"│ {'Time':<{W_TIME}} │"
    )
    print(header)
    print(_sep("├", "┼", "┤"))

    for i, r in enumerate(results):
        display_rows = _row(
            r["label"],
            r["answer"],
            r["input_tokens"],
            r["output_tokens"],
            r["cost_usd"],
            r["elapsed"],
        )
        for row_cols in display_rows:
            print(_data_row(row_cols))
        if i < len(results) - 1:
            print(_sep("├", "┼", "┤"))

    print(_sep("└", "┴", "┘"))


# ──────────────────────────────────────────────
# Summary helpers
# ──────────────────────────────────────────────

def print_final_summary(per_provider: dict[str, dict]):
    """Print total cost, tokens, time per provider and a recommendation."""
    print(f"\n{'═' * 82}")
    print("  FINAL SUMMARY")
    print(f"{'─' * 82}")

    W_LBL = 16
    W_TOT_COST = 10
    W_TOT_TOKS = 20
    W_TOT_TIME = 10

    header = (
        f"  {'Provider':<{W_LBL}}  "
        f"{'Total cost':<{W_TOT_COST}}  "
        f"{'Tokens (in / out)':<{W_TOT_TOKS}}  "
        f"{'Total time':<{W_TOT_TIME}}"
    )
    print(header)
    print(f"  {'─' * W_LBL}  {'─' * W_TOT_COST}  {'─' * W_TOT_TOKS}  {'─' * W_TOT_TIME}")

    rows_data = []
    for label, stats in per_provider.items():
        cost_str  = f"${stats['total_cost']:.4f}"
        toks_str  = f"{stats['total_in']:,} / {stats['total_out']:,}"
        time_str  = f"{stats['total_time']:.1f}s"
        print(
            f"  {label:<{W_LBL}}  "
            f"{cost_str:<{W_TOT_COST}}  "
            f"{toks_str:<{W_TOT_TOKS}}  "
            f"{time_str:<{W_TOT_TIME}}"
        )
        rows_data.append((label, stats))

    # Recommendation
    if len(rows_data) == 2:
        (lbl_a, s_a), (lbl_b, s_b) = rows_data
        cheaper = lbl_a if s_a["total_cost"] <= s_b["total_cost"] else lbl_b
        faster  = lbl_a if s_a["total_time"] <= s_b["total_time"] else lbl_b

        cost_ratio = max(s_a["total_cost"], s_b["total_cost"]) / max(
            min(s_a["total_cost"], s_b["total_cost"]), 1e-9
        )

        print(f"\n  Recommendation")
        print(f"  {'─' * 60}")
        print(f"  Cheapest  : {cheaper}  ({cost_ratio:.1f}x cheaper)")
        print(f"  Fastest   : {faster}")

        if cheaper == faster:
            print(f"\n  → {cheaper} wins on both cost AND speed.")
            print(
                "    Use Claude Haiku for high-volume or budget-constrained workloads.\n"
                "    Upgrade to GPT-4o or Claude Sonnet when deeper reasoning is needed."
            )
        else:
            print(
                f"\n  → Trade-off: {cheaper} is cheaper, {faster} is faster."
            )
            print(
                "    For most document Q&A workloads, cost is the dominant factor —\n"
                "    both models are fast enough for interactive use."
            )

    print()


# ──────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────

def run_experiment(pdf_path: str, domain: str):
    print(f"\n📄 Loading: {os.path.basename(pdf_path)}")
    text = extract_text(pdf_path)
    stats = get_doc_stats(text)
    print(
        f"📊 {stats['words']:,} words | ~{stats['estimated_tokens']:,} tokens | "
        f"{'✅ Safe' if stats['safe_for_context'] else '⚠️  Large document'}"
    )
    print(f"🌐 Domain: {domain} | Questions: {len(QUESTIONS)}\n")

    system_prompt = get_prompt(domain)

    clients = {
        p["key"]: (LLMClient(provider=p["key"]), p["label"])
        for p in PROVIDERS
    }

    # per_provider[label] = {total_cost, total_in, total_out, total_time}
    per_provider: dict[str, dict] = {
        p["label"]: {"total_cost": 0.0, "total_in": 0, "total_out": 0, "total_time": 0.0}
        for p in PROVIDERS
    }

    for q_idx, question in enumerate(QUESTIONS, start=1):
        print(f"  ── Question {q_idx}/{len(QUESTIONS)}: {question[:60]}...")
        results = []

        for p in PROVIDERS:
            client, label = clients[p["key"]]
            model = DEFAULT_MODELS[p["key"]]
            print(f"     {label}...", end=" ", flush=True)

            t0 = time.perf_counter()
            result = client.ask(
                question=question,
                context=text,
                system_prompt=system_prompt,
                model=model,
                temperature=0.1,
                max_tokens=MAX_TOKENS,
            )
            elapsed = time.perf_counter() - t0

            result["label"]   = label
            result["elapsed"] = elapsed
            results.append(result)

            # Accumulate per-provider totals
            s = per_provider[label]
            s["total_cost"] += result["cost_usd"]
            s["total_in"]   += result["input_tokens"]
            s["total_out"]  += result["output_tokens"]
            s["total_time"] += elapsed

            print(f"✓  ${result['cost_usd']:.4f} | {elapsed:.1f}s")

        print_question_table(q_idx, question, results)

    print_final_summary(per_provider)


def main():
    parser = argparse.ArgumentParser(description="Experiment 02 — Claude vs OpenAI side-by-side comparison")
    parser.add_argument("pdf_path", help="Path to a PDF file")
    parser.add_argument("--domain", choices=["legal", "technical", "hr"], default="legal")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"❌ File not found: {args.pdf_path}")
        sys.exit(1)

    run_experiment(args.pdf_path, args.domain)


if __name__ == "__main__":
    main()

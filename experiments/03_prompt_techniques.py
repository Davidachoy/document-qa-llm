"""
Experiment 03 — Prompt Techniques
===================================
Compares three prompting strategies on the same question over the same PDF:

  Zero-shot        : direct question, no examples or extra instructions
  Few-shot         : 2 Q&A examples provided before the real question
  Chain-of-Thought : "think step by step" instruction before answering

Shows responses side-by-side, token cost comparison (few-shot is more
expensive), and a subjective evaluation rubric to judge which technique
produced the most precise answer.

Usage:
    python experiments/03_prompt_techniques.py <pdf_path> [--domain legal|technical|hr]
    python experiments/03_prompt_techniques.py <pdf_path> --question "What is the duration?"
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

DEFAULT_QUESTION = "What are the main obligations or responsibilities of the parties involved?"

# Few-shot examples — synthetic Q&A pairs that teach the model
# the expected answer format (cites clause, stays grounded in document).
FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the duration of the contract?",
        "answer": (
            "According to Article 3, the contract lasts 12 months from the signing date, "
            "with automatic renewal for equal periods unless either party gives 30 days' "
            "written notice of non-renewal before the expiry date."
        ),
    },
    {
        "question": "Who is responsible for shipping costs?",
        "answer": (
            "Per Clause 5.2, the seller covers all shipping costs to the agreed delivery "
            "point. Return shipping is the buyer's responsibility, except when the product "
            "has manufacturing defects, in which case the seller bears the full cost."
        ),
    },
]

MAX_TOKENS = 400

# ──────────────────────────────────────────────
# Prompt builders
#
# LLMClient.ask() always produces:
#   "Document context:\n{context}\n\nQuestion: {question}"
#
# Each builder returns (question_str, context_str) so the final
# user message is assembled correctly by the client.
# ──────────────────────────────────────────────

def build_zero_shot(context: str, question: str) -> tuple[str, str]:
    """Plain question — no examples, no extra instructions."""
    return question, context


def build_few_shot(context: str, question: str) -> tuple[str, str]:
    """2 Q&A examples prepended to guide the model's answer format."""
    examples_block = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples_block += (
            f"Example —\n"
            f"Question: {ex['question']}\n"
            f"Answer: {ex['answer']}\n\n"
        )
    question_with_examples = (
        f"Below are examples showing the expected answer format:\n\n"
        f"{examples_block}"
        f"Using the same format, answer the following question:\n{question}"
    )
    return question_with_examples, context


def build_cot(context: str, question: str) -> tuple[str, str]:
    """Adds a step-by-step reasoning instruction before the question."""
    question_with_cot = (
        f"{question}\n\n"
        f"Before answering, think step by step:\n"
        f"  1. Which part of the document is relevant to this question?\n"
        f"  2. What does it say exactly about the topic?\n"
        f"  3. What is the precise, document-grounded answer?\n\n"
        f"Write your reasoning briefly, then your FINAL ANSWER clearly labeled."
    )
    return question_with_cot, context


TECHNIQUES = [
    {
        "key":   "zero_shot",
        "label": "Zero-shot",
        "desc":  "Direct question, no examples or special instructions.",
        "build": build_zero_shot,
    },
    {
        "key":   "few_shot",
        "label": "Few-shot",
        "desc":  "2 Q&A examples before the real question (more tokens, guided format).",
        "build": build_few_shot,
    },
    {
        "key":   "cot",
        "label": "Chain-of-Thought",
        "desc":  "Step-by-step reasoning instruction before the question.",
        "build": build_cot,
    },
]

# ──────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────

W_LABEL  = 18
W_ANSWER = 46
W_TOKENS = 13
W_COST   = 9
W_TIME   = 7


def _sep(left: str, mid: str, right: str, fill: str = "─") -> str:
    parts = [
        fill * (W_LABEL  + 2),
        fill * (W_ANSWER + 2),
        fill * (W_TOKENS + 2),
        fill * (W_COST   + 2),
        fill * (W_TIME   + 2),
    ]
    return left + mid.join(parts) + right


def _data_row(label: str, answer: str, tokens: str, cost: str, secs: str) -> str:
    return (
        f"│ {label:<{W_LABEL}} "
        f"│ {answer:<{W_ANSWER}} "
        f"│ {tokens:<{W_TOKENS}} "
        f"│ {cost:<{W_COST}} "
        f"│ {secs:<{W_TIME}} │"
    )


def print_comparison_table(question: str, results: list[dict]):
    wrapped_q = textwrap.fill(question, width=74, subsequent_indent="    ")
    print(f"\n{'═' * 92}")
    print(f"  Question: {wrapped_q}")
    print(_sep("┌", "┬", "┐"))

    header = (
        f"│ {'Technique':<{W_LABEL}} "
        f"│ {'Answer':<{W_ANSWER}} "
        f"│ {'Tokens in/out':<{W_TOKENS}} "
        f"│ {'Cost':<{W_COST}} "
        f"│ {'Time':<{W_TIME}} │"
    )
    print(header)
    print(_sep("├", "┼", "┤"))

    for i, r in enumerate(results):
        answer_lines = textwrap.wrap(r["answer"], width=W_ANSWER) or ["(empty)"]
        tokens_str = f"{r['input_tokens']:,}/{r['output_tokens']}"
        cost_str   = f"${r['cost_usd']:.4f}"
        time_str   = f"{r['elapsed']:.1f}s"

        for j, line in enumerate(answer_lines):
            if j == 0:
                print(_data_row(r["label"], line, tokens_str, cost_str, time_str))
            else:
                print(_data_row("", line, "", "", ""))

        if i < len(results) - 1:
            print(_sep("├", "┼", "┤"))

    print(_sep("└", "┴", "┘"))


def print_token_cost_breakdown(results: list[dict]):
    """Show token overhead of each technique relative to zero-shot."""
    print(f"\n  Token overhead per technique (vs Zero-shot)")
    print(f"  {'─' * 60}")

    base = next((r for r in results if r["key"] == "zero_shot"), None)
    if not base:
        return

    for r in results:
        extra_in   = r["input_tokens"]  - base["input_tokens"]
        extra_out  = r["output_tokens"] - base["output_tokens"]
        extra_cost = r["cost_usd"]      - base["cost_usd"]
        print(
            f"  {r['label']:<18}  "
            f"in: {extra_in:+,}  out: {extra_out:+,}  "
            f"cost: {extra_cost:+.4f} USD"
        )


def print_evaluation_rubric(results: list[dict]):
    """Subjective evaluation rubric — user fills this in mentally after reading answers."""
    labels = [r["label"] for r in results]
    col_w  = 14

    print(f"\n  Subjective evaluation rubric")
    print(f"  {'─' * (28 + col_w * len(labels))}")

    header_cols = "".join(f"{lbl:>{col_w}}" for lbl in labels)
    print(f"  {'Criterion':<28}{header_cols}")
    print(f"  {'─' * 28}" + "".join(["─" * col_w] * len(labels)))

    criteria = [
        "Cites clause/article?",
        "Answers exactly?",
        "Hallucinates data?",
        "Structured format?",
        "Reasoning visible?",
    ]
    blank = "[ ]"
    for c in criteria:
        row = "".join(f"{blank:>{col_w}}" for _ in labels)
        print(f"  {c:<28}{row}")

    print(f"\n  Legend: [✓] meets  [~] partial  [✗] does not meet")
    print(f"\n  General guidance:")
    print(f"  • Zero-shot       → simple factual retrieval, lowest cost")
    print(f"  • Few-shot        → when output format matters (lists, JSON, structured)")
    print(f"  • Chain-of-Thought → analytical, multi-step, or ambiguous questions")


# ──────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────

def run_experiment(pdf_path: str, domain: str, question: str):
    print(f"\n📄 Loading: {os.path.basename(pdf_path)}")
    text  = extract_text(pdf_path)
    stats = get_doc_stats(text)
    print(
        f"📊 {stats['words']:,} words | ~{stats['estimated_tokens']:,} tokens | "
        f"{'✅ Safe' if stats['safe_for_context'] else '⚠️  Large document'}"
    )

    model         = DEFAULT_MODELS["anthropic"]
    system_prompt = get_prompt(domain)
    client        = LLMClient(provider="anthropic")

    print(f"🤖 Model: {model}  |  Domain: {domain}")
    print(f"📋 Techniques: {', '.join(t['label'] for t in TECHNIQUES)}\n")

    results = []
    for tech in TECHNIQUES:
        print(f"  Running {tech['label']}...", end=" ", flush=True)
        question_str, context_str = tech["build"](text, question)

        t0 = time.perf_counter()
        result = client.ask(
            question=question_str,
            context=context_str,
            system_prompt=system_prompt,
            model=model,
            temperature=0.1,
            max_tokens=MAX_TOKENS,
        )
        elapsed = time.perf_counter() - t0

        result["key"]     = tech["key"]
        result["label"]   = tech["label"]
        result["desc"]    = tech["desc"]
        result["elapsed"] = elapsed
        results.append(result)
        print(f"✓  in={result['input_tokens']:,} out={result['output_tokens']} ${result['cost_usd']:.4f} {elapsed:.1f}s")

    print_comparison_table(question, results)
    print_token_cost_breakdown(results)
    print_evaluation_rubric(results)

    # Full answers for detailed reading
    print(f"\n{'═' * 92}")
    print("  FULL ANSWERS")
    for r in results:
        print(f"\n  ── {r['label']} ──")
        print(f"  {r['desc']}")
        print()
        for line in textwrap.wrap(r["answer"], width=84):
            print(f"  {line}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Experiment 03 — Prompt techniques comparison")
    parser.add_argument("pdf_path", help="Path to a PDF file")
    parser.add_argument("--domain",   choices=["legal", "technical", "hr"], default="legal")
    parser.add_argument("--question", default=DEFAULT_QUESTION,
                        help="Question to test across all three techniques")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"❌ File not found: {args.pdf_path}")
        sys.exit(1)

    run_experiment(args.pdf_path, args.domain, args.question)


if __name__ == "__main__":
    main()

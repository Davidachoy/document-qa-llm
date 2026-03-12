"""
Experiment 01 — Temperature Effects
====================================
Runs the same question at temperatures 0.0, 0.3, 0.7, 1.0 and compares
responses across three question types:
  1. Factual   → expected best at low temperature
  2. Summary   → expected best at medium temperature
  3. Analytical → expected best at higher temperature

Usage:
    python experiments/01_temperature_effects.py <pdf_path> [--domain legal|technical|hr]
"""

import argparse
import os
import sys
import textwrap

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_extractor import extract_text, get_doc_stats
from src.llm_client import LLMClient, DEFAULT_MODELS
from src.prompts import get_prompt
from src.cost_tracker import CostTracker

TEMPERATURES = [0.0, 0.3, 0.7, 1.0]

TEST_QUESTIONS = [
    {
        "type": "Factual",
        "question": "What specific dates, amounts, or numbers are mentioned in this document?",
        "expected_best_temp": 0.0,
        "reason": "Factual retrieval requires deterministic output — low temperature reduces hallucination.",
    },
    {
        "type": "Summary",
        "question": "Summarize the main purpose and key points of this document in 3-4 sentences.",
        "expected_best_temp": 0.3,
        "reason": "Summaries benefit from slight creativity to vary phrasing, but must stay accurate.",
    },
    {
        "type": "Analytical",
        "question": "What are the potential risks or implications for the parties involved in this document?",
        "expected_best_temp": 0.7,
        "reason": "Analytical reasoning benefits from diverse perspectives and exploratory thinking.",
    },
]

DIVIDER = "─" * 72
WRAP_WIDTH = 65


def print_response_table(question_type: str, question: str, results: list[dict]):
    print(f"\n{'═' * 72}")
    print(f"  Question type : {question_type}")
    print(f"  Question      : {textwrap.fill(question, width=60, subsequent_indent='                  ')}")
    print(DIVIDER)

    for r in results:
        wrapped = textwrap.fill(r["answer"], width=WRAP_WIDTH, subsequent_indent="    ")
        print(f"\n  🌡  temp={r['temperature']:.1f}")
        print(f"  {wrapped}")
        print(f"  🔢 {r['input_tokens']:,} in / {r['output_tokens']:,} out  |  💰 ${r['cost_usd']:.6f}")
        print(DIVIDER)


def print_observation(question_meta: dict, results: list[dict]):
    expected = question_meta["expected_best_temp"]
    print(f"\n  📝 Observation ({question_meta['type']}):")
    print(f"     Expected best temperature : {expected}")
    print(f"     Why: {textwrap.fill(question_meta['reason'], width=60, subsequent_indent='          ')}")


def run_experiment(pdf_path: str, domain: str):
    print(f"\n📄 Loading: {os.path.basename(pdf_path)}")
    text = extract_text(pdf_path)
    stats = get_doc_stats(text)
    print(
        f"📊 {stats['words']:,} words | ~{stats['estimated_tokens']:,} tokens | "
        f"{'✅ Safe' if stats['safe_for_context'] else '⚠️  Large'}"
    )

    client = LLMClient(provider="anthropic")
    model = DEFAULT_MODELS["anthropic"]
    system_prompt = get_prompt(domain)
    tracker = CostTracker()

    print(f"🤖 Model: {model}  |  Domain: {domain}")
    print(f"🌡  Temperatures: {TEMPERATURES}")
    print(f"📋 Questions: {len(TEST_QUESTIONS)}\n")

    for q_meta in TEST_QUESTIONS:
        results = []

        for temp in TEMPERATURES:
            print(f"  Running [{q_meta['type']}] at temp={temp:.1f}...", end=" ", flush=True)
            result = client.ask(
                question=q_meta["question"],
                context=text,
                system_prompt=system_prompt,
                model=model,
                temperature=temp,
                max_tokens=300,
            )
            result["temperature"] = temp
            results.append(result)
            tracker.log_call(model, result["input_tokens"], result["output_tokens"], result["cost_usd"])
            print(f"✓  (${result['cost_usd']:.6f})")

        print_response_table(q_meta["type"], q_meta["question"], results)
        print_observation(q_meta, results)

    print(f"\n{'═' * 72}")
    print("  💰 Experiment Total")
    print(DIVIDER)
    tracker.print_summary()
    print()


def main():
    parser = argparse.ArgumentParser(description="Experiment 01 — Temperature effects on LLM responses")
    parser.add_argument("pdf_path", help="Path to a PDF in sample_docs/")
    parser.add_argument("--domain", choices=["legal", "technical", "hr"], default="legal")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"❌ File not found: {args.pdf_path}")
        sys.exit(1)

    run_experiment(args.pdf_path, args.domain)


if __name__ == "__main__":
    main()

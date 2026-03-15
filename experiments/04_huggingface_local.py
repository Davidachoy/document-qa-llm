"""
Experiment 04 — HuggingFace Local (generative) vs Claude API
=============================================================
Runs the same document through a local instruction-following model
(no API key, $0 cost) and through Claude Haiku, comparing quality and speed.

Local model: Qwen/Qwen2.5-1.5B-Instruct
  - Modern generative model (~3 GB, downloads once to ~/.cache/huggingface/)
  - 32k token context window — no chunking needed for typical documents
  - Supports chat-style messages (system / user / assistant roles)
  - Uses pipeline("text-generation"), the officially supported API in v5

Why this model?
  transformers v5 removed the legacy pipeline tasks "question-answering"
  and "summarization" because modern instruction-tuned models do those tasks
  better via text-generation + a chat prompt.

Flow:
  1. Extract PDF text
  2. Auto-detect best device (MPS on Apple Silicon, CUDA if available, else CPU)
  3. Load Qwen2.5-1.5B-Instruct via pipeline("text-generation")
  4. Run 3 Q&A questions  — local model vs Claude Haiku
  5. Generate summary     — local model vs Claude Haiku
  6. Print side-by-side tables + overall cost/speed report

Usage:
    python experiments/04_huggingface_local.py <pdf_path> [--domain legal|technical|hr]
    python experiments/04_huggingface_local.py <pdf_path> --hf-model Qwen/Qwen2.5-0.5B-Instruct
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

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

QUESTIONS = [
    "What is the duration or term of this document?",
    "Who are the parties involved in this document?",
    "What are the main obligations or responsibilities?",
]

HF_MAX_NEW_TOKENS   = 300
CLAUDE_MAX_TOKENS   = 300

# Safety truncation: Qwen2.5-1.5B has a 32k token window.
# At ~4 chars/token that is ~128k chars. Most PDFs are well under this.
HF_MAX_CONTEXT_CHARS = 120_000

# ──────────────────────────────────────────────
# HuggingFace — load
# ──────────────────────────────────────────────

def _best_device() -> str:
    """Pick MPS (Apple Silicon) > CUDA > CPU automatically."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_hf_pipeline(model_id: str):
    """
    Load an instruction-following model via pipeline("text-generation").

    This is the officially recommended API in transformers v5 for QA,
    summarization, and any text-to-text task. The model receives a list
    of chat messages and returns the assistant reply.
    """
    try:
        from transformers import pipeline
    except ImportError:
        print("❌ Missing dependency. Run: pip install transformers torch")
        sys.exit(1)

    device = _best_device()
    print(f"  Device  : {device}")
    print(f"  Model   : {model_id}")
    print(f"  (first run downloads ~3 GB to ~/.cache/huggingface/)")

    t0 = time.perf_counter()
    pipe = pipeline(
        "text-generation",
        model=model_id,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Loaded in {elapsed:.1f}s\n")
    return pipe


# ──────────────────────────────────────────────
# HuggingFace — inference
# ──────────────────────────────────────────────

def _hf_chat(pipe, system: str, user: str) -> tuple[str, float]:
    """
    Send a chat message to the local pipeline and return (answer, elapsed).

    The pipeline returns:
        result[0]["generated_text"]  →  full message list (system + user + assistant)
        result[0]["generated_text"][-1]["content"]  →  just the assistant reply
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    t0 = time.perf_counter()
    # Pass generation settings via GenerationConfig to avoid the deprecation
    # warning raised when max_new_tokens conflicts with the model's built-in
    # generation_config (which may set max_length=20 by default).
    from transformers import GenerationConfig
    gen_cfg = GenerationConfig(max_new_tokens=HF_MAX_NEW_TOKENS)
    result = pipe(messages, generation_config=gen_cfg)
    elapsed = time.perf_counter() - t0

    reply = result[0]["generated_text"][-1]["content"]
    return reply.strip(), elapsed


def hf_ask(pipe, question: str, context: str, system_prompt: str) -> dict:
    """Ask a question about the document using the local generative model."""
    truncated = context[:HF_MAX_CONTEXT_CHARS]
    user_msg  = f"Document:\n{truncated}\n\nQuestion: {question}"

    answer, elapsed = _hf_chat(pipe, system_prompt, user_msg)
    return {
        "answer":   answer,
        "elapsed":  elapsed,
        "cost_usd": 0.0,
    }


def hf_summarize(pipe, context: str, system_prompt: str) -> dict:
    """Summarize the document using the local generative model."""
    truncated = context[:HF_MAX_CONTEXT_CHARS]
    user_msg  = (
        f"Document:\n{truncated}\n\n"
        f"Summarize the main purpose and key points of this document in 3-4 sentences."
    )

    summary, elapsed = _hf_chat(pipe, system_prompt, user_msg)
    return {
        "summary":  summary,
        "elapsed":  elapsed,
        "cost_usd": 0.0,
    }


# ──────────────────────────────────────────────
# Claude helpers
# ──────────────────────────────────────────────

def claude_ask(client, question: str, context: str, system_prompt: str, model: str) -> dict:
    t0 = time.perf_counter()
    result = client.ask(
        question=question,
        context=context,
        system_prompt=system_prompt,
        model=model,
        temperature=0.1,
        max_tokens=CLAUDE_MAX_TOKENS,
    )
    result["elapsed"] = time.perf_counter() - t0
    return result


def claude_summarize(client, context: str, system_prompt: str, model: str) -> dict:
    t0 = time.perf_counter()
    result = client.ask(
        question="Summarize the main purpose and key points of this document in 3-4 sentences.",
        context=context,
        system_prompt=system_prompt,
        model=model,
        temperature=0.1,
        max_tokens=200,
    )
    result["elapsed"] = time.perf_counter() - t0
    return result


# ──────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────

W_RUNNER = 20
W_ANSWER = 48
W_COST   = 9
W_TIME   = 8


def _sep(left: str, mid: str, right: str, fill: str = "─") -> str:
    parts = [
        fill * (W_RUNNER + 2),
        fill * (W_ANSWER + 2),
        fill * (W_COST   + 2),
        fill * (W_TIME   + 2),
    ]
    return left + mid.join(parts) + right


def _data_row(runner: str, answer: str, cost: str, secs: str) -> str:
    return (
        f"│ {runner:<{W_RUNNER}} "
        f"│ {answer:<{W_ANSWER}} "
        f"│ {cost:<{W_COST}} "
        f"│ {secs:<{W_TIME}} │"
    )


def print_qa_table(q_index: int, question: str, rows: list[dict]):
    wrapped_q = textwrap.fill(question, width=76, subsequent_indent="    ")
    print(f"\n{'═' * 92}")
    print(f"  Q{q_index}: {wrapped_q}")
    print(_sep("┌", "┬", "┐"))
    header = (
        f"│ {'Runner':<{W_RUNNER}} "
        f"│ {'Answer':<{W_ANSWER}} "
        f"│ {'Cost':<{W_COST}} "
        f"│ {'Time':<{W_TIME}} │"
    )
    print(header)
    print(_sep("├", "┼", "┤"))

    for i, r in enumerate(rows):
        answer_lines = textwrap.wrap(r["answer"], width=W_ANSWER) or ["(no answer)"]
        cost_str = f"${r['cost_usd']:.4f}"
        time_str = f"{r['elapsed']:.2f}s"

        for j, line in enumerate(answer_lines):
            if j == 0:
                print(_data_row(r["runner"], line, cost_str, time_str))
            else:
                print(_data_row("", line, "", ""))

        if i < len(rows) - 1:
            print(_sep("├", "┼", "┤"))

    print(_sep("└", "┴", "┘"))


def print_summary_comparison(hf_result: dict, claude_result: dict, hf_label: str):
    print(f"\n{'═' * 92}")
    print("  DOCUMENT SUMMARY — side by side")
    print(f"{'─' * 92}")

    hf_lines     = textwrap.wrap(hf_result["summary"],     width=43)
    claude_lines = textwrap.wrap(claude_result["answer"],   width=43)
    n = max(len(hf_lines), len(claude_lines))
    hf_lines     += [""] * (n - len(hf_lines))
    claude_lines += [""] * (n - len(claude_lines))

    left_hdr  = f"{hf_label}  $0.0000  {hf_result['elapsed']:.2f}s"
    right_hdr = f"Claude Haiku  ${claude_result['cost_usd']:.4f}  {claude_result['elapsed']:.2f}s"
    print(f"  {left_hdr:<45}  {right_hdr}")
    print(f"  {'─' * 43}  {'─' * 43}")
    for l, r in zip(hf_lines, claude_lines):
        print(f"  {l:<43}  {r:<43}")


def print_overall_report(qa_rows_all: list[list[dict]], hf_label: str):
    print(f"\n{'═' * 92}")
    print("  OVERALL REPORT")
    print(f"{'─' * 92}")

    totals: dict[str, dict] = {}
    for rows in qa_rows_all:
        for r in rows:
            key = r["runner"]
            if key not in totals:
                totals[key] = {"time": 0.0, "cost": 0.0, "calls": 0}
            totals[key]["time"]  += r["elapsed"]
            totals[key]["cost"]  += r["cost_usd"]
            totals[key]["calls"] += 1

    W_R = 22
    print(f"  {'Runner':<{W_R}}  {'Calls':>6}  {'Total time':>12}  {'Total cost':>12}  {'Avg/call':>10}")
    print(f"  {'─' * W_R}  {'─' * 6}  {'─' * 12}  {'─' * 12}  {'─' * 10}")
    for runner, t in totals.items():
        avg = t["time"] / t["calls"] if t["calls"] else 0
        print(
            f"  {runner:<{W_R}}  {t['calls']:>6}  "
            f"{t['time']:>11.2f}s  "
            f"${t['cost']:>11.4f}  "
            f"{avg:>9.2f}s"
        )

    print(f"\n  Trade-off analysis")
    print(f"  {'─' * 65}")
    print(f"  {hf_label} (local generative):")
    print(f"    ✓ Zero cost — no API key, no usage fees")
    print(f"    ✓ Data stays 100% local — ideal for sensitive documents")
    print(f"    ✓ Generative — reasons, synthesizes, not just extractive")
    print(f"    ✓ 32k token context — handles most documents without chunking")
    print(f"    ✗ Slower than API on CPU (~10–60s per answer vs ~1–3s)")
    print(f"    ✗ Smaller model — less world knowledge and reasoning depth")
    print(f"  Claude Haiku (API):")
    print(f"    ✓ Stronger reasoning and instruction-following")
    print(f"    ✓ 200k token context")
    print(f"    ✓ Fast — ~1–3s per answer")
    print(f"    ✗ Requires API key, costs per token")
    print(f"    ✗ Data sent to Anthropic's servers")
    print()
    print(f"  When to use local:")
    print(f"    Sensitive documents (legal, medical, financial), offline use,")
    print(f"    high-volume batch processing where API cost is prohibitive.")
    print(f"  When to use Claude:")
    print(f"    Complex analytical questions, speed matters, or quality is critical.")
    print()


# ──────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────

def run_experiment(pdf_path: str, domain: str, hf_model: str):
    print(f"\n📄 Loading: {os.path.basename(pdf_path)}")
    text  = extract_text(pdf_path)
    stats = get_doc_stats(text)
    print(
        f"📊 {stats['words']:,} words | ~{stats['estimated_tokens']:,} tokens | "
        f"{'✅ Fits in context' if stats['safe_for_context'] else '⚠️  Large document'}"
    )

    system_prompt = get_prompt(domain)
    hf_label      = hf_model.split("/")[-1]   # e.g. "Qwen2.5-1.5B-Instruct"

    # ── Load local model ────────────────────────────────────
    print(f"\n🤗 Loading local HuggingFace model...")
    pipe = load_hf_pipeline(hf_model)

    # ── Load Claude ──────────────────────────────────────────
    print("🤖 Connecting to Claude API...")
    claude_model = DEFAULT_MODELS["anthropic"]
    try:
        claude_client = LLMClient(provider="anthropic")
        print(f"   Model: {claude_model}  |  Domain: {domain}\n")
    except EnvironmentError as e:
        print(f"   ⚠️  Claude unavailable: {e}")
        claude_client = None

    # ── Q&A comparison ───────────────────────────────────────
    print(f"{'─' * 92}")
    print(f"  Running {len(QUESTIONS)} questions × 2 runners...")
    print(f"{'─' * 92}")

    qa_rows_all = []

    for q_idx, question in enumerate(QUESTIONS, start=1):
        rows = []

        # Local model
        print(f"  Q{q_idx} — {hf_label}...", end=" ", flush=True)
        hf_res = hf_ask(pipe, question, text, system_prompt)
        rows.append({
            "runner":   hf_label,
            "answer":   hf_res["answer"],
            "cost_usd": 0.0,
            "elapsed":  hf_res["elapsed"],
        })
        print(f"✓  {hf_res['elapsed']:.2f}s")

        # Claude
        if claude_client:
            print(f"  Q{q_idx} — Claude Haiku...  ", end=" ", flush=True)
            cl_res = claude_ask(claude_client, question, text, system_prompt, claude_model)
            rows.append({
                "runner":   "Claude Haiku",
                "answer":   cl_res["answer"],
                "cost_usd": cl_res["cost_usd"],
                "elapsed":  cl_res["elapsed"],
            })
            print(f"✓  {cl_res['elapsed']:.2f}s  ${cl_res['cost_usd']:.4f}")

        print_qa_table(q_idx, question, rows)
        qa_rows_all.append(rows)

    # ── Summary comparison ───────────────────────────────────
    print(f"\n{'─' * 92}")
    print("  Generating summaries...")
    print(f"{'─' * 92}")

    print(f"  {hf_label}...", end=" ", flush=True)
    hf_summ = hf_summarize(pipe, text, system_prompt)
    print(f"✓  {hf_summ['elapsed']:.2f}s")

    claude_summ = None
    if claude_client:
        print(f"  Claude Haiku...  ", end=" ", flush=True)
        claude_summ = claude_summarize(claude_client, text, system_prompt, claude_model)
        print(f"✓  {claude_summ['elapsed']:.2f}s  ${claude_summ['cost_usd']:.4f}")

        qa_rows_all.append([
            {"runner": hf_label,       "elapsed": hf_summ["elapsed"],     "cost_usd": 0.0},
            {"runner": "Claude Haiku", "elapsed": claude_summ["elapsed"], "cost_usd": claude_summ["cost_usd"]},
        ])
        print_summary_comparison(hf_summ, claude_summ, hf_label)

    else:
        print(f"\n  Local summary:\n  {hf_summ['summary']}")

    # ── Overall report ───────────────────────────────────────
    print_overall_report(qa_rows_all, hf_label)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 04 — HuggingFace local generative model vs Claude API"
    )
    parser.add_argument("pdf_path", help="Path to a PDF file")
    parser.add_argument("--domain",   choices=["legal", "technical", "hr"], default="legal")
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL,
                        help=f"HuggingFace model ID (default: {DEFAULT_HF_MODEL})")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"❌ File not found: {args.pdf_path}")
        sys.exit(1)

    run_experiment(args.pdf_path, args.domain, args.hf_model)


if __name__ == "__main__":
    main()

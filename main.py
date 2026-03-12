import argparse
import os
import sys

from src.pdf_extractor import extract_text, get_doc_stats
from src.prompts import get_prompt, VALID_DOMAINS
from src.llm_client import LLMClient, DEFAULT_MODELS
from src.cost_tracker import CostTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Document Q&A — ask questions about a PDF")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--domain",
        choices=sorted(VALID_DOMAINS),
        default="legal",
        help="Document domain (default: legal)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider (default: anthropic)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load PDF ──────────────────────────────────────────────────────────────
    if not os.path.exists(args.pdf_path):
        print(f"❌ File not found: {args.pdf_path}")
        sys.exit(1)

    print(f"\n📄 Loading document: {os.path.basename(args.pdf_path)}")
    text = extract_text(args.pdf_path)
    stats = get_doc_stats(text)

    safe_icon = "✅ Safe" if stats["safe_for_context"] else "⚠️  Large"
    print(
        f"📊 Stats: {stats['words']:,} words | "
        f"~{stats['estimated_tokens']:,} tokens | {safe_icon}"
    )
    if not stats["safe_for_context"]:
        print("⚠️  Document exceeds 150k tokens — responses may be incomplete.")

    # ── Init client & tracker ─────────────────────────────────────────────────
    try:
        client = LLMClient(provider=args.provider)
    except EnvironmentError as e:
        print(f"❌ {e}")
        sys.exit(1)

    model = DEFAULT_MODELS[args.provider]
    system_prompt = get_prompt(args.domain)
    tracker = CostTracker()

    print(f"🤖 Provider: {args.provider} ({model})")
    print(f"📁 Domain  : {args.domain}")
    print("─" * 42)
    print('Type your question, "stats" for session summary, or "exit" to quit.\n')

    # ── Interactive loop ──────────────────────────────────────────────────────
    while True:
        try:
            question = input("❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            question = "exit"

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("\n── Session complete ──")
            tracker.print_summary()
            break

        if question.lower() == "stats":
            tracker.print_summary()
            print()
            continue

        try:
            result = client.ask(
                question=question,
                context=text,
                system_prompt=system_prompt,
                model=model,
            )
        except Exception as e:
            print(f"❌ Error: {e}\n")
            continue

        tracker.log_call(
            model=model,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            cost=result["cost_usd"],
        )

        print(f"\n💬 Answer:\n{result['answer']}")
        print(
            f"\n🔢 Tokens: {result['input_tokens']:,} in / {result['output_tokens']:,} out"
            f" | 💰 ${result['cost_usd']:.6f}\n"
        )
        print("─" * 42)


if __name__ == "__main__":
    main()

import fitz  # pymupdf
import tiktoken


def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF, with page separators."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        pages.append(f"--- Page {i} ---\n{page.get_text()}")
    doc.close()
    return "\n\n".join(pages)


def get_doc_stats(text: str, model: str = "claude-opus-4-6") -> dict:
    """Return character, word, and token counts for a text string.

    Uses cl100k_base encoding as a universal approximation.
    safe_for_context is True when estimated tokens are under 150k.
    """
    # cl100k_base is a good approximation for both Claude and OpenAI models
    enc = tiktoken.get_encoding("cl100k_base")
    estimated_tokens = len(enc.encode(text))

    return {
        "chars": len(text),
        "words": len(text.split()),
        "estimated_tokens": estimated_tokens,
        "safe_for_context": estimated_tokens < 150_000,
    }


if __name__ == "__main__":
    import sys
    import os

    # Use a PDF path from CLI args, or print usage if none provided
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <pdf_path>")
        print("\nDemo with synthetic text:")
        sample = "This is a sample contract.\n\nClause 1: Lorem ipsum dolor sit amet."
        stats = get_doc_stats(sample)
        print(f"  Text: {sample[:60]!r}...")
        print(f"  Stats: {stats}")
        sys.exit(0)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    text = extract_text(pdf_path)
    stats = get_doc_stats(text)

    print(f"Pages extracted: {text.count('--- Page')}")
    print(f"Stats: {stats}")
    print(f"\nFirst 300 characters:\n{text[:300]}")

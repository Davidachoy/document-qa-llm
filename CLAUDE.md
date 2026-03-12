# document-qa-llm

Q&A engine over PDF documents using LLM APIs (Claude & OpenAI).

## Project Overview

- **Purpose:** Ask questions about PDF documents using LLM APIs
- **Language:** Python 3.11+
- **Phase:** Early development (structure only, no logic yet)

## Key Dependencies

- `anthropic` — Claude API
- `openai` — OpenAI API
- `pymupdf` — PDF parsing
- `python-dotenv` — Environment variable management
- `tiktoken` — Token counting

## Environment Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add API keys to .env
```

Required environment variables (see `.env.example`):
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`

## Project Structure

```
src/          # Core source code (Q&A engine logic)
experiments/  # Exploratory scripts
use_cases/
  legal/      # Legal document Q&A
  technical/  # Technical documentation Q&A
  hr/         # HR document Q&A
sample_docs/  # PDF samples (gitignored)
```

## Notes

- `sample_docs/` is gitignored — add PDFs locally for testing
- Default to `claude-opus-4-6` for Claude API calls
- Use streaming for any requests with large inputs or outputs

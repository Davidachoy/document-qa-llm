# document-qa-llm

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-orange?logo=anthropic)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white)

Q&A engine over PDF documents using Claude and OpenAI APIs. Part of an AI Engineer learning path focused on LLM fundamentals, prompt engineering, and real cost/quality trade-offs across providers and model sizes.

---

## Features

- **Multi-provider** — Claude Haiku and GPT-4o-mini from the same interface, same cost table
- **Multi-domain** — separate system prompts and CLIs for legal, technical, and HR documents
- **Real-time cost tracking** — every call logs input/output tokens and USD cost
- **4 comparative experiments** — temperature effects, provider comparison, prompt techniques, local vs API
- **Local model support** — run `Qwen2.5-1.5B-Instruct` entirely offline via HuggingFace (no API key, $0)

---

## Project structure

```
document-qa-llm/
│
├── src/                          # Core engine
│   ├── llm_client.py             # Unified Claude + OpenAI client with retry logic
│   ├── pdf_extractor.py          # PDF text extraction + token estimation
│   ├── prompts.py                # Domain-specific system prompts
│   └── cost_tracker.py           # Per-session token and cost aggregation
│
├── experiments/
│   ├── 01_temperature_effects.py # How temperature affects factual vs analytical answers
│   ├── 02_claude_vs_openai.py    # Same 5 questions, both providers, side-by-side table
│   ├── 03_prompt_techniques.py   # Zero-shot vs few-shot vs chain-of-thought
│   └── 04_huggingface_local.py   # Local Qwen2.5-1.5B vs Claude Haiku
│
├── use_cases/
│   ├── legal/main_legal.py       # Contracts — suggested questions + Costa Rica law context
│   ├── technical/main_technical.py # API docs — code block rendering in terminal
│   └── hr/main_hr.py             # Employee handbooks — friendly tone + HR footer
│
├── sample_docs/                  # PDFs for testing (gitignored)
│   ├── legal/contract.pdf        # Generated sample: 3-page software services contract
│   ├── technical/
│   └── hr/
│
├── scripts/
│   └── make_sample_pdf.py        # Generates contract.pdf using stdlib only (no deps)
│
├── main.py                       # General-purpose CLI (any domain, any provider)
├── requirements.txt
├── .env.example
└── CLAUDE.md
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/davidachoy/document-qa-llm
cd document-qa-llm

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and add your keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   OPENAI_API_KEY=sk-...

# 5. Run
python main.py sample_docs/legal/contract.pdf --domain legal
```

---

## Basic usage

```bash
# General CLI — any domain, any provider
python main.py <pdf> --domain legal|technical|hr --provider anthropic|openai

# Domain-specific CLIs (suggested questions + formatted output)
python use_cases/legal/main_legal.py     sample_docs/legal/contract.pdf
python use_cases/technical/main_technical.py sample_docs/technical/api_spec.pdf
python use_cases/hr/main_hr.py           sample_docs/hr/handbook.pdf
```

**Example session:**

```
════════════════════════════════════════════════════════════════════
  Legal Assistant — Contracts (Costa Rica)
──────────────────────────────────────────────────────────────────
  Document : contract.pdf
  Words    : 892  |  Tokens: ~1,847
  Context  : fits in context
  Model    : claude-haiku-4-5-20251001
════════════════════════════════════════════════════════════════════

  Suggested questions:

  [1]  What are the parties to the contract and how are they identified?
  [2]  What is the duration of the contract and what is the start date?
  [3]  What are the main obligations of each party?
  [4]  What are the conditions and grounds for termination or rescission?

  > 2
  -> What is the duration of the contract and what is the start date?

──────────────────────────────────────────────────────────────────
  Answer:

  Per Article 2.1, the contract commences on March 1, 2026 and runs
  for twelve (12) months, expiring on February 28, 2027. Under
  Article 2.2, it renews automatically for successive 12-month
  periods unless either party provides written notice of
  non-renewal at least 30 days before the expiry date.

  tokens: 1,923 in / 94 out  | $0.0011  |  1.4s
──────────────────────────────────────────────────────────────────
```

---

## Experiments

Each script runs standalone and produces a formatted comparison table.

### 01 — Temperature effects
```bash
python experiments/01_temperature_effects.py sample_docs/legal/contract.pdf
```
Tests temps `0.0 / 0.3 / 0.7 / 1.0` on three question types (factual, summary, analytical). **Finding:** factual retrieval is most reliable at `0.0`; analytical questions get more diverse reasoning at `0.7`, but accuracy doesn't improve.

### 02 — Claude vs OpenAI
```bash
python experiments/02_claude_vs_openai.py sample_docs/legal/contract.pdf
```
Runs 5 identical questions through Claude Haiku and GPT-4o-mini. Prints a side-by-side table with tokens, cost, and latency per provider.

```
  Question 1: What is the duration or term of this document?
┌──────────────────┬──────────────────────────────┬───────────────┬──────────┬─────────┐
│ Provider         │ Answer                       │ Tokens in/out │ Cost     │ Time    │
├──────────────────┼──────────────────────────────┼───────────────┼──────────┼─────────┤
│ Claude Haiku     │ Per Article 2.1, the contract│ 1,923/94      │ $0.0011  │ 1.4s    │
├──────────────────┼──────────────────────────────┼───────────────┼──────────┼─────────┤
│ GPT-4o-mini      │ The contract lasts 12 months │ 1,901/87      │ $0.0003  │ 0.9s    │
└──────────────────┴──────────────────────────────┴───────────────┴──────────┴─────────┘
```

### 03 — Prompt techniques
```bash
python experiments/03_prompt_techniques.py sample_docs/legal/contract.pdf
```
Compares **zero-shot**, **few-shot** (2 examples), and **chain-of-thought** on the same question. Includes a token overhead table (few-shot adds ~400 input tokens) and a fillable evaluation rubric.

### 04 — HuggingFace local vs Claude
```bash
python experiments/04_huggingface_local.py sample_docs/legal/contract.pdf
```
Runs `Qwen/Qwen2.5-1.5B-Instruct` locally (no API key) against Claude Haiku. First run downloads ~3 GB; subsequent runs use cache. Automatically uses MPS on Apple Silicon, CUDA if available, otherwise CPU.

```bash
# Smaller/faster local model option:
python experiments/04_huggingface_local.py contract.pdf --hf-model Qwen/Qwen2.5-0.5B-Instruct
```

---

## Use cases

| CLI | Domain | Special feature |
|-----|--------|-----------------|
| `use_cases/legal/main_legal.py` | Contracts | Costa Rica law context, cites exact clauses |
| `use_cases/technical/main_technical.py` | API docs, manuals | Renders ` ``` ` code blocks in a terminal box |
| `use_cases/hr/main_hr.py` | Employee handbooks | Friendly tone, HR disclaimer on every answer |

**Technical CLI code block rendering:**
```
  ╭─ python ─────────────────────────────────────────────────────╮
  │ import requests                                              │
  │ headers = {"Authorization": f"Bearer {token}"}              │
  │ r = requests.get("https://api.example.com/v1/data",         │
  │                  headers=headers)                            │
  ╰──────────────────────────────────────────────────────────────╯
```

---

## Cost comparison

Results from running 5 questions on a ~1,900 token document (3-page contract):

| Model | Provider | Input rate | Output rate | 5 questions total |
|-------|----------|-----------|------------|-------------------|
| claude-haiku-4-5 | Anthropic | $1.00/M | $5.00/M | ~$0.006 |
| gpt-4o-mini | OpenAI | $0.15/M | $0.60/M | ~$0.001 |
| Qwen2.5-1.5B (local) | HuggingFace | $0 | $0 | $0.000 |
| claude-sonnet-4-6 | Anthropic | $3.00/M | $15.00/M | ~$0.020 |

**Takeaway:** GPT-4o-mini is ~6x cheaper than Claude Haiku for the same document size. Both are fast enough for interactive use (~1–2s). The local Qwen model is free but ~10–40x slower on CPU and produces shallower answers on complex questions.

---

## What I learned

- **Token counting matters before you hit send.** A 10-page contract is ~4k tokens. A 100-page agreement is ~40k. Knowing this upfront lets you choose the right model and avoid surprise costs.

- **System prompt domain matters more than model size.** A targeted domain prompt (legal, technical, HR) consistently produced more precise, citable answers than a generic prompt on a larger model.

- **Few-shot is expensive for extraction tasks.** Adding 2 examples added ~400 input tokens per call — a 20% cost increase — with minimal quality gain on factual retrieval. CoT helped more on analytical questions.

- **Local models are real competition for simple QA.** `Qwen2.5-1.5B` answered direct factual questions correctly on our contract. It struggled with multi-clause reasoning where Claude synthesizes across sections.

---

## Next

Phase 2 — *document-qa-rag*: chunking strategies, vector stores, and retrieval-augmented generation for documents that exceed the context window.

> Link will be added when the repo goes public.

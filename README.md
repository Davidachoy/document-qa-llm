# document-qa-llm

Q&A engine over PDF documents using LLM APIs (Claude & OpenAI). Part of an AI Engineer learning path focused on LLM fundamentals, prompt engineering, and token management.

## Project Structure

```
document-qa-llm/
├── src/                  # Core source code
├── experiments/          # Exploratory scripts and notebooks
├── use_cases/
│   ├── legal/            # Legal document Q&A use cases
│   ├── technical/        # Technical documentation Q&A
│   └── hr/               # HR document Q&A
├── sample_docs/          # Sample PDFs (gitignored)
│   ├── legal/
│   ├── technical/
│   └── hr/
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

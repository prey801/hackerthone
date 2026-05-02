# Support Triage Agent

A terminal-based AI agent that triages support tickets across HackerRank, Claude, and Visa using a hybrid retrieval system and Gemini for structured responses.

## Architecture

```
corpus/loader.py        → Loads + chunks data/ markdown files into segments
retrieval/embedder.py   → sentence-transformers wrapper (all-MiniLM-L6-v2)
retrieval/bm25_index.py → BM25 keyword index
retrieval/vector_index.py → FAISS dense vector index
retrieval/retriever.py  → Hybrid search (40% BM25 + 60% Dense), top-K chunks
triage/input_validator.py → Prompt injection & invalid input detection
triage/domain_detector.py → Detect HackerRank / Claude / Visa from company or keywords
triage/risk_assessor.py → Rule-based escalation triggers (fraud, legal, security, etc.)
triage/responder.py     → Gemini API → structured JSON output (pydantic schema)
pipeline.py             → Orchestration: validate → detect → retrieve → risk → respond
run_batch.py            → Batch runner → output.csv
main.py                 → Interactive terminal mode
```

## Setup

### Prerequisites
- Python 3.9+
- `uv` (recommended) or `pip`
- A Gemini API key from [aistudio.google.com](https://aistudio.google.com)

### Install dependencies
```bash
# From the code/ directory:
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Configure API key
```bash
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY
```

## Usage

### Quick Start (Recommended)
You can run the entire batch pipeline (including environment setup and dependency installation) with a single command from the project root:
```bash
./run.sh
```

### Manual Interactive mode (single ticket)
```bash
cd code/
source .venv/bin/activate
python main.py
```

### Manual Batch mode (generate output.csv)
```bash
cd code/
source .venv/bin/activate
python run_batch.py \
  --input  ../support_tickets/support_tickets.csv \
  --output ../support_tickets/output.csv \
  --data   ../data/
```

## Output Format

Each row in `output.csv` contains:

| Column         | Values |
|----------------|--------|
| `status`       | `replied` or `escalated` |
| `product_area` | support category / domain area |
| `response`     | grounded user-facing answer |
| `justification`| concise routing rationale |
| `request_type` | `product_issue`, `feature_request`, `bug`, or `invalid` |

## Design Decisions

- **Hybrid Retrieval**: BM25 (40%) + FAISS dense vectors (60%) for best of keyword + semantic matching.
- **Rule-based risk first**: Escalation triggers run *before* the LLM, so high-risk cases (fraud, GDPR, security) are always escalated regardless of LLM output.
- **Structured output**: Pydantic schema passed to Gemini ensures valid JSON with correct field values every time.
- **Injection detection**: Multilingual patterns (English + French/Spanish) catch adversarial inputs like ticket #21.
- **Determinism**: `random.seed(42)` and `temperature=0.0` ensure reproducible outputs.
- **No hallucinations**: The system prompt explicitly forbids outside knowledge; if retrieval is low-confidence (score < 0.35), the ticket is escalated.

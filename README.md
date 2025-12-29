# 10-K Risk Analysis Platform

RAG-powered search and comparison tool for SEC 10-K filings. Built for equity research analysts who need to quickly answer "What changed?" between annual reports.

![Light Mode](https://via.placeholder.com/800x400?text=Screenshot+Coming+Soon)

## Features

- **Semantic Search** — Find relevant risk disclosures across 5 companies and 5 years using natural language queries
- **Compare Years** — Side-by-side comparison of what changed between fiscal years (same company)
- **Compare Companies** — See how Company A discusses a topic vs Company B (same year)
- **LLM Synthesis** — AI-generated summaries with citations [1][2][3] for verification
- **Export** — Download results as Word or PDF for client deliverables
- **Dark Mode** — Easy on the eyes during late-night research sessions

## Covered Companies

| Company | Ticker | Years |
|---------|--------|-------|
| Meta | META | FY2020-2024 |
| Apple | AAPL | FY2020-2024 |
| Alphabet | GOOG | FY2021-2024 |
| Microsoft | MSFT | FY2020-2024 |
| Amazon | AMZN | FY2020-2024 |

**Sections indexed:** Item 1A (Risk Factors) and Item 7 (MD&A)

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/meta-sec.git
cd meta-sec
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
# Edit .env with your keys:
#   OPENROUTER_API_KEY - required for LLM features (https://openrouter.ai)
#   SEC_API_KEY - only needed if re-extracting data (https://sec-api.io)
```

### 3. Run the app

```bash
source .env
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

### Search Mode
Enter a query like "AI regulation" or "China supply chain" to find relevant excerpts across all companies and years. Results show:
- Hit counts by fiscal year (trend signal)
- Relevance scores
- LLM-generated summary with citations

### Compare Years Mode
Select two fiscal years for the same company to see what changed. The LLM identifies:
- New disclosures
- Removed language
- Wording changes (e.g., "significant" → "intense and increasing")

### Compare Companies Mode
Select two companies and a topic to see side-by-side how each discusses the same risk. Useful for competitive analysis.

### Keyboard Shortcuts
- `/` — Focus search box
- `Esc` — Unfocus
- `Enter` — Submit query

## Tech Stack

- **Frontend:** Streamlit
- **Embeddings:** BGE-base-en-v1.5 (via sentence-transformers)
- **Vector Store:** FAISS (IndexFlatIP)
- **LLM:** Google Gemini 2.0 Flash (via OpenRouter)
- **Data Source:** SEC EDGAR (via sec-api.io)

## Project Structure

```
meta-sec/
├── app.py                 # Main Streamlit application
├── scripts/
│   ├── extract_risk_factors.py   # Pull filings from SEC API
│   ├── clean_data.py             # Clean HTML artifacts
│   ├── chunk_data.py             # Semantic chunking
│   └── embed_and_index.py        # Build FAISS index
├── tests/
│   ├── eval_retrieval.py         # Retrieval quality evaluation
│   └── run_eval.py               # LLM-as-judge evaluation
├── sec_corpus/            # Extracted 10-K text by company
├── vector_store/          # FAISS index + metadata (gitignored)
├── assets/
│   └── custom.css         # UI styling
└── docs/
    └── client_interview_notes.md  # Requirements gathering
```

## Data Pipeline

To refresh or add new companies:

```bash
# 1. Extract from SEC (requires SEC_API_KEY)
python scripts/extract_risk_factors.py

# 2. Clean HTML artifacts
python scripts/clean_data.py

# 3. Chunk with metadata
python scripts/chunk_data.py

# 4. Build vector index
python scripts/embed_and_index.py
```

## Development

This project was built using a client roleplay methodology to gather realistic requirements before coding. See `docs/client_interview_notes.md` for the full requirements gathering process.

Key design decisions:
- **Comparison-aware search** over structural diff matching (see CLAUDE.md for rationale)
- **Verbatim excerpts** with citations, not paraphrased summaries
- **Recall over precision** — better to surface too much than miss something

## License

MIT

## Acknowledgments

- SEC filings sourced via [sec-api.io](https://sec-api.io)
- LLM inference via [OpenRouter](https://openrouter.ai)

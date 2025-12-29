# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG system for searching 10-K filings (Risk Factors + MD&A) across Big Tech companies (META, AAPL, GOOG, MSFT, AMZN) from FY2020-2024. Built for equity research analysts to answer "What changed?" between filings. Features semantic search with hit counts by year, comparison-aware LLM synthesis, and Perplexity-style citations.

**Live deployment:** https://sec-risk-analysis-rag.streamlit.app

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy .env.example and add your keys
cp .env.example .env
# Edit .env with your API keys:
#   SEC_API_KEY - required for extraction (https://sec-api.io)
#   OPENROUTER_API_KEY - required for LLM features (https://openrouter.ai)

# Load environment variables before running
source .env  # or use python-dotenv
```

## API Keys

**NEVER hardcode API keys in source files.** All keys go in `.env` (which is gitignored).

When modifying code that uses API keys:
- Read from `os.environ.get("KEY_NAME")` or `st.secrets.get("KEY_NAME")` for Streamlit Cloud
- Keys in `.env`: `SEC_API_KEY`, `OPENROUTER_API_KEY`

## Commands

```bash
# Full pipeline (run in order from project root)
python scripts/extract_risk_factors.py    # Extract from SEC (needs API key)
python scripts/clean_data.py              # Clean HTML artifacts
python scripts/chunk_data.py              # Chunk with metadata
python scripts/embed_and_index.py         # Build FAISS index

# Run search UI
streamlit run app.py

# Run evaluation (uses DeepEval with cached test cases)
python tests/run_eval.py                  # Quick eval with summary
python tests/eval_retrieval.py            # Full eval with question generation
```

## Project Structure

```
sec-risk-analysis/
├── app.py                  # Main Streamlit application
├── scripts/                # Data pipeline scripts
│   ├── extract_risk_factors.py
│   ├── clean_data.py
│   ├── chunk_data.py
│   └── embed_and_index.py
├── tests/                  # Evaluation scripts
│   ├── eval_retrieval.py   # Full DeepEval with synthetic question generation
│   ├── run_eval.py         # Quick eval using cached test cases
│   └── eval_test_cases.json
├── sec_corpus/             # Source and processed text files by company
│   ├── META/
│   ├── AAPL/
│   ├── GOOG/
│   ├── MSFT/
│   └── AMZN/
├── vector_store/           # FAISS index and metadata
├── assets/                 # UI assets (custom.css)
├── .streamlit/             # Streamlit Cloud configuration
│   └── config.toml
└── packages.txt            # System dependencies for Streamlit Cloud
```

## Architecture

**Data Pipeline:**
1. `extract_risk_factors.py` → queries sec-api for 10-K filings, extracts Item 1A to `sec_corpus/{TICKER}/FY{year}_risk_factors.txt`
2. `clean_data.py` → decodes HTML entities, removes artifacts → `sec_corpus/{TICKER}/cleaned/`
3. `chunk_data.py` → semantic chunking (lead sentences, bullet groups, paragraphs with overlap) → `sec_corpus/{TICKER}/chunked/all_chunks.json`
4. `embed_and_index.py` → BGE embeddings + FAISS IndexFlatIP → `vector_store/`

**Chunk types:** lead_sentence (high retrieval value), bullet_group (with parent_context), paragraph (split if >512 tokens)

**Application (`app.py`):**
- **Search Mode:** Semantic search across all years, hit counts by year (trend signal), LLM summary with [1][2][3] citations
- **Compare Mode:** Search both years, LLM compares excerpts directly to identify new topics, removed topics, and wording changes
- **LLM Integration:** Google Gemini 2.0 Flash via OpenRouter for synthesis
- **Design Pattern:** Perplexity-style — search + synthesize + cite sources

**Evaluation:** `eval_retrieval.py` - DeepEval with synthetic questions, ContextualRecall/Precision metrics

## Confidence Thresholds (in app.py)

- `HIGH_CONFIDENCE = 0.75` - search relevance score for "strong match"
- `MEDIUM_CONFIDENCE = 0.55` - below this = weak match

## SEC Filing Notes

- 10-K filed in early year X = fiscal year X-1 (filed Feb 2024 → FY2023)
- Meta ticker changed FB → META in June 2022

## Client Requirements Summary

- **Primary use case:** Year-over-year comparison ("What changed?")
- **Accuracy philosophy:** "Don't miss things" — recall > precision
- **Output:** Verbatim excerpts with citations, not paraphrased summaries
- **Target users:** Equity research analysts preparing for earnings season

## Design Decision: Why Not Side-by-Side Chunk Comparison?

We initially built a feature that matched chunks between years by semantic similarity and labeled them NEW/MODIFIED/REMOVED. This was **removed** because:

1. **Semantic similarity ≠ structural identity** — Two chunks being 80% similar doesn't mean they're the same section that was edited. Companies can reorganize risk factors entirely between years.
2. **Misleading labels** — Calling something "MODIFIED" implies it's the same paragraph that changed, but we were just matching similar-sounding text.
3. **Client trust** — False confidence is worse than no feature. The client emphasized: "if I get burned once by a 'high confidence' hallucination, I'll stop trusting the system entirely."

**Current approach:** LLM compares excerpts directly and identifies changes (including subtle wording shifts like "significant" → "intense and increasing"). This is more honest and actually catches the hard cases analysts care about.

## Evaluation Results (2025-12-28)

LLM-as-Judge evaluation (Gemini) — **Verdict: LEGITIMATE & HIGH QUALITY**

| Test | Query | Result |
|------|-------|--------|
| Retrieval Quality | "AI" | Found specific FY2024 risk language about AI investments |
| Semantic Inference | "TikTok" | Found competitor risks even though TikTok not mentioned by name |
| Comparison Logic | "Efficiency" FY23 vs FY22 | Detected "virtual/augmented reality" → "Reality Labs" terminology shift |

Key findings:
- App performs substantially better than keyword search
- LLM summaries are faithful to source text without hallucination
- Semantic inference correctly handles implicit references (TikTok → competitor risks)
- Comparison mode detects real terminology and risk escalation changes

## Deployment

Deployed on Streamlit Cloud at https://sec-risk-analysis-rag.streamlit.app

Configuration files:
- `.streamlit/config.toml` — theme and server settings
- `packages.txt` — system dependencies (libomp-dev for faiss-cpu)
- Secrets managed via Streamlit Cloud dashboard (OPENROUTER_API_KEY)

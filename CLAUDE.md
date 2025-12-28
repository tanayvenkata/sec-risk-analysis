# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG system for searching Meta's 10-K Risk Factors (Item 1A) from FY2020-2024. Built for equity research analysts to answer "What changed?" between filings. Features semantic search with hit counts by year, comparison-aware LLM synthesis, and Perplexity-style citations.

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
- Read from `os.environ.get("KEY_NAME")`
- If removing a hardcoded key, move it to `.env` first so the app keeps working
- Keys in `.env`: `SEC_API_KEY`, `OPENROUTER_API_KEY`

## Commands

```bash
# Full pipeline (run in order)
python extract_risk_factors.py    # Extract from SEC (needs API key)
python clean_data.py              # Clean HTML artifacts
python chunk_data.py              # Chunk with metadata
python embed_and_index.py         # Build FAISS index

# Run search UI
streamlit run app.py

# Run retrieval evaluation
python eval_retrieval.py
```

## Architecture

**Data Pipeline:**
1. `extract_risk_factors.py` → queries sec-api for 10-K filings, extracts Item 1A to `sec_corpus/META/FY{year}_risk_factors.txt`
2. `clean_data.py` → decodes HTML entities, removes artifacts → `sec_corpus/META/cleaned/`
3. `chunk_data.py` → semantic chunking (lead sentences, bullet groups, paragraphs with overlap) → `sec_corpus/META/chunked/all_chunks.json`
4. `embed_and_index.py` → BGE embeddings + FAISS IndexFlatIP → `vector_store/`

**Chunk types:** lead_sentence (high retrieval value), bullet_group (with parent_context), paragraph (split if >512 tokens)

**Application (`app.py`):**
- **Search Mode:** Semantic search across all years, hit counts by year (trend signal), LLM summary with [1][2][3] citations
- **Compare Mode:** Search both years, LLM compares excerpts directly to identify new topics, removed topics, and wording changes
- **LLM Integration:** Google Gemini 2.0 Flash via OpenRouter for synthesis
- **Design Pattern:** Perplexity-style — search + synthesize + cite sources

**Evaluation:** `eval_retrieval.py` - DeepEval with synthetic questions, ContextualRecall/Precision metrics

## Key Files

- `vector_store/faiss_index.bin` - FAISS index
- `vector_store/metadata.pkl` - chunk metadata for retrieval
- `sec_corpus/META/chunked/all_chunks.json` - all processed chunks
- `eval_test_cases.json` - cached evaluation test cases
- `docs/client_interview_notes.md` - client requirements and feedback

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
- **Timeline:** Mid-January 2025 for stable version (Big Tech 10-Ks drop late January)

## Design Decision: Why Not Side-by-Side Chunk Comparison?

We initially built a feature that matched chunks between years by semantic similarity and labeled them NEW/MODIFIED/REMOVED. This was **removed** because:

1. **Semantic similarity ≠ structural identity** — Two chunks being 80% similar doesn't mean they're the same section that was edited. Meta can reorganize risk factors entirely between years.
2. **Misleading labels** — Calling something "MODIFIED" implies it's the same paragraph that changed, but we were just matching similar-sounding text.
3. **Client trust** — False confidence is worse than no feature. The client said: "if I get burned once by a 'high confidence' hallucination, I'll stop trusting the system entirely."

**Current approach:** LLM compares excerpts directly and identifies changes (including subtle wording shifts like "significant" → "intense and increasing"). This is more honest and actually catches the hard cases analysts care about.

## Evaluation Results (2025-12-28)

LLM-as-Judge evaluation (Gemini) — **Verdict: LEGITIMATE & HIGH QUALITY**

| Test | Query | Result |
|------|-------|--------|
| Retrieval Quality | "AI" | ✅ Found specific FY2024 risk language about AI investments |
| Semantic Inference | "TikTok" | ✅ Found competitor risks even though TikTok not mentioned by name |
| Comparison Logic | "Efficiency" FY23 vs FY22 | ✅ Detected "virtual/augmented reality" → "Reality Labs" terminology shift |

Key findings:
- App performs substantially better than keyword search
- LLM summaries are faithful to source text without hallucination
- Semantic inference correctly handles implicit references (TikTok → competitor risks)
- Comparison mode detects real terminology and risk escalation changes

# High-Level Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌──────────────┐         Queries like:                                   │
│    │   Analyst    │         • "What's new in FY2024 vs FY2023?"             │
│    │  (Web UI)    │         • "Show me all AI regulation mentions"          │
│    └──────┬───────┘         • "How has China risk evolved?"                 │
│           │                                                                  │
└───────────┼─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │                      Query Router                                 │     │
│    │   • Classifies query type (diff / search / comparison / trend)   │     │
│    │   • Routes to appropriate retrieval strategy                      │     │
│    └──────────────────────────┬───────────────────────────────────────┘     │
│                               │                                              │
│           ┌───────────────────┼───────────────────┐                         │
│           ▼                   ▼                   ▼                         │
│    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
│    │  Diff       │    │  Semantic   │    │  Hybrid     │                   │
│    │  Engine     │    │  Search     │    │  Search     │                   │
│    │             │    │             │    │             │                   │
│    │ YoY compare │    │ Topic/fact  │    │ Keyword +   │                   │
│    │ at paragraph│    │ retrieval   │    │ semantic    │                   │
│    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                   │
│           │                  │                  │                           │
│           └──────────────────┼──────────────────┘                           │
│                              ▼                                              │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │                    Response Generator                             │     │
│    │   • Formats excerpts with citations                              │     │
│    │   • Adds [NEW], [MODIFIED], [REMOVED] labels                     │     │
│    │   • Assigns confidence scores                                     │     │
│    │   • Suggests manual verification when uncertain                   │     │
│    └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌────────────────────┐         ┌────────────────────┐                    │
│    │   Vector Store     │         │   Document Store   │                    │
│    │                    │         │                    │                    │
│    │ • Chunked paragraphs│        │ • Raw text files   │                    │
│    │ • Embeddings        │◄──────►│ • Metadata (year,  │                    │
│    │ • Section metadata  │        │   section, company)│                    │
│    │                    │         │                    │                    │
│    └────────────────────┘         └────────────────────┘                    │
│              ▲                              ▲                                │
│              │                              │                                │
│              └──────────────┬───────────────┘                               │
│                             │                                                │
│    ┌────────────────────────┴───────────────────────────────────────────┐   │
│    │                     Ingestion Pipeline                              │   │
│    │   • Extracts Item 1A from 10-K filings                             │   │
│    │   • Chunks by paragraph, preserves section headers                 │   │
│    │   • Generates embeddings                                            │   │
│    │   • Stores with (company, year, section) metadata                  │   │
│    └────────────────────────────────────────────────────────────────────┘   │
│                             ▲                                                │
│                             │                                                │
└─────────────────────────────┼───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL DATA SOURCE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌──────────────────┐                                                     │
│    │   SEC EDGAR      │     via sec-api.io                                  │
│    │   (10-K Filings) │     • Query API: find filings                       │
│    │                  │     • Extractor API: pull Item 1A                   │
│    └──────────────────┘                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

| Component | Purpose | Technology Options |
|-----------|---------|-------------------|
| **Web UI** | Query interface for analysts | Streamlit, Gradio, or simple React app |
| **Query Router** | Classify intent, route to strategy | LLM classifier or rule-based |
| **Diff Engine** | YoY paragraph comparison | Text diffing + semantic similarity |
| **Semantic Search** | Topic/concept retrieval | Vector similarity search |
| **Vector Store** | Store embeddings + metadata | Pinecone, Chroma, or Qdrant |
| **Document Store** | Raw text + structured metadata | Filesystem or PostgreSQL |
| **Ingestion Pipeline** | ETL from SEC → indexed chunks | Python scripts (existing) |

---

## Data Flow

### Ingestion (Batch)
```
SEC EDGAR → sec-api.io → Raw Text Files → Chunking → Embeddings → Vector Store
                              ↓
                         Metadata JSON
```

### Query (Real-time)
```
User Query → Router → Retrieval Strategy → Relevant Chunks → LLM Formatting → Response
                                                  ↓
                                          (with citations)
```

---

## Critical Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| **Chunking granularity** | Paragraph-level | Citations need specificity |
| **Preserve section headers** | Yes | "Government Regulation section" in citations |
| **Retrieval tuning** | High recall, accept noise | False negatives are unacceptable |
| **Diff as first-class feature** | Yes | #1 use case is YoY comparison |
| **Confidence scoring** | Required | Analyst needs to know when to verify |

---

## Technical Decisions (2025-12-27)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Diff approach | Semantic diff | Client OK with slower; catches paraphrasing |
| Embedding model | Open source (TBD) | Test locally first, decide later |
| Hosting | Local demo | Test and iterate before deployment |
| Scope | Meta only, FY2020-2024 | Prove on one company first |

## Client Sign-Off (2025-12-27)

- [x] Client confirms architecture matches their mental model
- [x] Query types in scope agreed (diff, search, comparison, trend)
- [x] Response format approved (quotes + citations + confidence)
- [x] Hosting approach decided → Local demo first
- [x] Embedding model → Open source, test locally
- [x] MVP scope defined → Meta only

**Key insight from client:** Side-by-side diff showing old vs. new language is the "killer feature"


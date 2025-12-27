# Risk Factor Analysis Tool — Proposed Design

**Prepared for:** Equity Research Team
**Date:** 2025-12-27

---

## How It Works

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│   You Ask a     │ ───► │   System Finds  │ ───► │   You Get       │
│   Question      │      │   Relevant      │      │   Excerpts +    │
│                 │      │   Sections      │      │   Citations     │
│                 │      │                 │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘

     Web interface         Searches across          Direct quotes,
     (no coding)           all years indexed        source labeled
```

---

## What You Can Ask

| Query Type | Example | What You Get |
|------------|---------|--------------|
| **What changed?** | "What's new in FY2024 vs FY2023?" | Side-by-side diff with [NEW], [MODIFIED], [REMOVED] labels |
| **Topic search** | "Find all mentions of AI regulation" | Every relevant excerpt across all years |
| **Keyword lookup** | "Show me everywhere Meta mentions Apple" | Exact matches with surrounding context |
| **Trend over time** | "How has China risk language evolved?" | Year-by-year comparison with changes highlighted |

---

## What Responses Look Like

Every response includes:

> **[NEW in FY2024]**
> *"We face regulatory scrutiny regarding our use of artificial intelligence, including pending legislation in the European Union that may impose significant requirements on AI systems..."*
>
> — **META FY2024 10-K, Item 1A, Government Regulation section**
> **Confidence: High** (phrase not present in FY2023)

---

> **[MODIFIED from FY2023]**
> *"Our Reality Labs segment has incurred significant operating losses..."*
> *(Previously: "We expect our Reality Labs segment to continue to grow...")*
>
> — **META FY2024 10-K, Item 1A, Strategic Risks section**
> **Confidence: High**

---

> **[UNCERTAIN — Verify Manually]**
> Found 3 possible references to antitrust risk. Excerpts below may overlap with general regulatory language:
> - Excerpt 1: ... [link to section]
> - Excerpt 2: ... [link to section]
> - Excerpt 3: ... [link to section]

---

## Design Principles

Based on our conversation:

| Principle | How It's Built |
|-----------|----------------|
| **Don't miss things** | System tuned for high recall — may surface extra matches, but won't skip relevant ones |
| **Always cite sources** | Every excerpt tagged with company, year, and section |
| **Be honest about uncertainty** | Confidence scores on every response; "verify manually" when unsure |
| **Show actual language** | Direct quotes, never paraphrased summaries |
| **No coding required** | Simple web interface — type question, get answer |

---

## Data Source

- **Source:** SEC EDGAR (official filings)
- **What's indexed:** Item 1A (Risk Factors) from 10-K annual reports
- **Coverage:** Meta FY2020–FY2024 (expandable to other companies)

---

## Confirmation Checklist

| Item | Confirmed |
|------|-----------|
| Year-over-year diff is the primary use case | ✓ "That's the 80% use case" |
| Direct quotes preferred over summaries | ✓ "I need to quote these in my notes" |
| False positives are acceptable; false negatives are not | ✓ "Surface too much, not too little" |
| Confidence scores + "verify manually" prompts are helpful | ✓ "Sets the right expectation" |
| Web interface (no scripts/terminal) is required | ✓ "Non-negotiable for adoption" |
| Starting with Meta only is fine for POC | ✓ "Prove it on one first" |
| Target: testable by mid-January | ✓ |

---

## Scope Decisions

| Question | Answer |
|----------|--------|
| Multi-company | Meta only for POC. Next: Apple, Google, Microsoft, Amazon |
| Historical depth | FY2020-2024 is perfect (COVID → Meta rebrand → metaverse → AI pivot) |
| Access | Just the primary analyst for now; team demo if successful |

---

## Key Insight

> "That 'MODIFIED' example — where you display the old language alongside the new? That's *exactly* what I want. If you can reliably do that side-by-side comparison, that's the killer feature."

**Side-by-side diff with old vs. new language is the highest-value feature.**

---

*Confirmed 2025-12-27. Proceeding to build.*

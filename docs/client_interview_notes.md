# Client Interview Notes

**Date:** 2025-12-27
**Stakeholder:** Senior Equity Research Associate
**Topic:** RAG Strategy for Meta 10-K Risk Factors

---

## Problem Statement

During earnings season, analyst has 3-4 hours per company to digest press release, call, and 10-K/10-Q. The 10-K alone is 150+ pages. Core need: **"What changed?"**

Risk factors (Item 1A) is where companies quietly disclose material changes:
- New litigation
- Regulatory threats
- Supply chain issues
- Competitive dynamics

Current workflow: Manual Ctrl+F with two browser tabs. Painful, error-prone, definitely misses things.

---

## User Requirements

### Query Types (Priority Order)

| Query Type | Example | Frequency |
|------------|---------|-----------|
| Year-over-year diff | "What's new in FY2024 vs FY2023 risk factors?" | Every filing |
| Topic search | "Find all mentions of AI regulation" | Weekly |
| Cross-company comparison | "Which companies mention Apple as a risk?" | Monthly |
| Specific fact lookup | "What does Meta say about EU antitrust?" | Ad hoc |
| Trend analysis | "How has Meta's China risk language evolved since 2020?" | Quarterly |

### Response Format Requirements

Must include:
1. **Direct quotes** — actual text, not paraphrased
2. **Source citation** — company, fiscal year, section
3. **Context** — is this new, modified, or removed?
4. **Confidence indicator** — certain or needs verification?

Example output format:
> **[NEW in FY2024]** *"We face regulatory scrutiny regarding our use of artificial intelligence..."*
> — META FY2024 10-K, Item 1A, Government Regulation section
> **Confidence: High** (exact phrase not present in FY2023)

### Escalation Behavior

When uncertain, system should:
- "I found partial matches in these sections — review manually"
- "This topic spans multiple risk categories — here are all relevant excerpts"
- "No direct matches found, but related language exists here"

**Never** return "I don't know" without pointing somewhere.

---

## Accuracy Philosophy

> "I don't need the system to be *right*, I need it to **not miss things**."

| Type | Tolerance | Rationale |
|------|-----------|-----------|
| False positives | OK | "This might be new" → analyst verifies |
| False negatives | BAD | Missing a DOJ subpoena disclosure = problem |

**Citations are non-negotiable.** Every claim must point to exact section and year.

---

## Success Metrics

**Primary:** Time saved
- Current: 4-6 hours per company during annual filing season
- Target: 30 minutes to "what changed" with high confidence nothing missed

**Secondary:** Catch rate
- Cross-check manually for a quarter
- Measure: How often did system surface something analyst would have missed?

---

## Non-Functional Requirements

| Requirement | Spec | Notes |
|-------------|------|-------|
| Latency (single company) | < 30 seconds | Not real-time; desk research |
| Latency (cross-company) | Few minutes OK | If output is comprehensive |
| Concurrency | 2-3 users | Small team |
| Compliance | None special | Public SEC filings only |

---

## Constraints

### Infrastructure
- Uses Bloomberg, FactSet, Microsoft Office
- **Not a developer** — needs web interface or simple app
- No Python scripts or server management

### Privacy
- Queries aren't sensitive (public filings)
- Don't expose query history to competitors
- Basic access controls sufficient

### Budget
| Tier | Approval Path |
|------|---------------|
| < $500/mo | Can expense directly |
| $500-2000/mo | Need to show PM clear ROI |
| > $2000/mo | Team decision, months to approve |

**POC:** Free — needs to prove value first

---

## Timeline

- Big Tech 10-K filings: Late January - Early February
- **Target:** Testable POC by mid-January
- Doesn't need polish — core retrieval must work

---

## RAG Strategy Implications

Based on interview, key design decisions:

| Decision | Direction | Rationale |
|----------|-----------|-----------|
| **Chunking** | Paragraph-level with section headers preserved | Need to cite specific locations |
| **Retrieval priority** | Recall > Precision | False negatives are unacceptable |
| **Output format** | Verbatim excerpts, not summaries | "I am the analysis" |
| **Diff capability** | Core feature | YoY comparison is #1 use case |
| **Confidence scoring** | Required | Must flag uncertainty |
| **Multi-year indexing** | Required | Trend analysis across 5 years |


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

---

## Follow-up Questions (2025-12-27)

Clarifying questions on YoY diff implementation, UI, and confidence/escalation behavior.

### 1. Workflow Preference

**Query-focused, not document browsing.**

> "I'm not going to sit there and browse a 40-page diff document hoping something catches my eye. I come in with a question — 'what changed about AI risk?' or 'what's new this year?' — and I want answers to *that*."

**Exception:** Start of earnings season — may want a one-time "give me everything that changed" dump, but organized by topic/category, not raw diff.

### 2. Visual Format

**For POC:** List view with tags is acceptable. Functional is fine.

**Ideal (v2+):** Side-by-side comparison — old language on left, new on right, changes highlighted.

> "That's what I'm doing manually right now with two browser windows."

**Guidance:** Leave UI to developer discretion for v1. Get retrieval right first, polish UI later.

### 3. Confidence Indicator

**Definition by trust level, not technical implementation:**

| Level | Meaning | Analyst Action |
|-------|---------|----------------|
| High | Can quote in research note without verifying | Trust it |
| Medium | Should glance at source to confirm | Quick check |
| Low / Verify | This is a lead, not an answer | Manual review required |

> "How you calculate that is your call. Just make sure 'high confidence' actually means something — if I get burned once by a 'high confidence' hallucination, I'll stop trusting the system entirely."

### 4. Escalation Behavior

**Trigger:** Developer discretion (low similarity, ambiguous matches, etc.)

**Output format when uncertain:**

> "I found 3 sections that might be relevant to your query about [X]. I'm not confident these are complete or correctly categorized. Here they are — verify against the source."

Then provide excerpts with links. Don't synthesize if not confident — point to the right places.

**Key principle:** False confidence is the worst outcome.

> "The worst outcome is false confidence — telling me something is 'new' when it's actually just rephrased, or missing something because the similarity score was below some threshold. When in doubt, show me more and flag the uncertainty."

---

## Updated Design Decisions (Post Follow-up)

| Decision | Direction | Rationale |
|----------|-----------|-----------|
| **YoY Diff** | Query-focused, not document browse | User comes with a question, wants answers to that |
| **UI (POC)** | List view with tags | Functional first, polish later |
| **UI (Future)** | Side-by-side comparison | Mirrors current manual workflow |
| **Confidence** | Raw scores, user interprets | Simplified from 3-tier; let analyst judge |
| **Escalation** | Show more + flag uncertainty | Never false confidence; over-retrieve if unsure |
| **Generation** | Minimal for POC | "I am the analysis" — verbatim excerpts preferred |

---

## Implementation Progress (2025-12-27)

### Completed

| Feature | Status | Notes |
|---------|--------|-------|
| Data extraction | ✅ | FY2020-2024 Item 1A extracted |
| Chunking | ✅ | Semantic chunks with metadata |
| Embeddings + FAISS | ✅ | BGE model, cosine similarity |
| Search mode | ✅ | Works well, raw scores shown |
| Compare mode | ✅ | Query-focused, global search then filter by year |
| Change tags | ✅ | NEW/MODIFIED/REMOVED/UNCHANGED |
| Show All Changes | ✅ | Checkbox for full document diff |

### UX Feedback (to address in v2)

**Problem:** Compare mode results lack context. User gets "thrown into the deep end" with raw chunks.

Current experience:
- See chunk tagged as MODIFIED
- No context about what section, what topic, why it matters
- Have to Ctrl+F in PDF to understand surrounding context

**Potential solutions:**
1. **Add section context** — Show parent section header, not just chunk
2. **Summarization layer** — LLM provides brief intro: "This section about AI regulation was modified..."
3. **Visual diff** — Highlight exact words that changed between versions
4. **Grouping** — Group changes by topic/section rather than flat list

**Client's preference:** TBD — need to validate which approach they want

### Remaining Work

| Feature | Priority | Complexity | Notes |
|---------|----------|------------|-------|
| **LLM summary/context** | High | Medium | Walk user through findings with context |
| **Trend analysis view** | Medium | Medium | "How has X evolved 2020-2024?" |
| **Better MODIFIED display** | Medium | Low | Inline diff highlighting |
| **Query routing** | Low | Medium | Auto-detect search vs compare intent |
| **Cross-company** | Low | Low | Data scope issue, not code |

---

## POC Demo Feedback (2025-12-27)

Client reviewed screenshots and demo of Search and Compare modes.

### Search Mode Feedback

**What works:**
- Summary with [1][2][3] citations — "exactly what I asked for"
- Can scan summary, drill into excerpts to verify/quote
- "Patterns/Trends" section saves synthesis time
- Score guide helps calibrate trust per result
- Correctly found AI regulation only in FY2023-2024 (not hallucinating matches for earlier years)

**Concerns:**
- Summaries may be too long for quick lookups
- **Request:** Collapse/expand option for summaries

### Compare Mode Feedback

> "This is the killer feature."

**What works:**
- At-a-glance breakdown (3 New / 2 Modified / 0 Removed / 12 Unchanged) — "exactly what I need"
- "Key Changes" section explaining *why each change matters* — "beyond what I asked for, actually useful"
- Multi-year gap comparison (FY2024 vs FY2021) for longer-arc analysis
- Tabs let user focus on what they care about

**Questions from client:**
- How is "MODIFIED" detected? Semantic similarity?
- If Meta completely rephrased a risk, would it catch that or mark as "removed + new"?
- Side-by-side diff — is that available? ("That's what I really want for MODIFIED items")

### Threshold Feedback

> "70% = modified, 90% = unchanged seems reasonable as a starting point."

Client wants to test edge cases before finalizing. Requested example of MODIFIED item showing both versions.

### Workflow Feedback

- **Compare mode = 80% use case**
- **Search mode = ad hoc lookups** ("did they mention TikTok anywhere?")

### Real Queries Client Would Run

| Query | Purpose |
|-------|---------|
| "Apple iOS ATT" or "App Tracking Transparency" | Huge deal in FY2021-2022 |
| "FTC" or "antitrust" | Regulatory risk |
| "layoffs" or "workforce reduction" | "Year of efficiency" language |
| "China" | Geopolitical risk |

### Priority for Next Features (Client Ranked)

| Priority | Feature | Notes |
|----------|---------|-------|
| **#1** | Side-by-side diff for MODIFIED | "The missing piece" — old vs new language |
| **#2** | Show ALL changes without topic filter | "Everything that changed FY2023 → FY2024" |
| **#3** | Export to Word/PDF | Paste into research notes |
| Lower | Trend view across 5 years | Nice to have |
| Lower | Cross-company | Phase 2 |

### Timeline

- Big Tech 10-Ks drop **late January**
- Client wants to use this for Meta's FY2024 filing analysis
- **Target: mid-January for stable version**

### Overall Assessment

> "This is further along than I expected. The Compare mode is doing 80% of what I need. Get me that side-by-side diff and this is usable."

---

## Updated Priorities (Post Demo)

| Feature | Priority | Status |
|---------|----------|--------|
| Side-by-side diff for MODIFIED | **P0** | ~~Not started~~ → Pivoted (see below) |
| Collapsible summaries | P1 | Not started |
| Show ALL changes (no query) | P1 | ✅ Already built (checkbox) |
| Export to Word/PDF | P2 | Not started |
| Test with real queries (ATT, FTC, China, layoffs) | P1 | ✅ Completed |

---

## Architectural Pivot (2025-12-28)

### The Problem We Discovered

After implementing side-by-side diff for MODIFIED items, we identified a **fundamental flaw** in the approach:

**Semantic similarity between chunks ≠ structural identity.**

When we matched chunks by embedding similarity (70-90% = "MODIFIED"), we were saying "these chunks sound similar" — but that doesn't mean they're the same paragraph that was edited. Meta can:
- Reorganize risk factors entirely between years
- Split one risk into multiple paragraphs
- Consolidate multiple risks into one
- Move content between sections

The "MODIFIED" label implied structural alignment that didn't exist. This could mislead the analyst into thinking they were seeing the same section edited, when in reality they were seeing two different parts of the document that happened to discuss similar topics.

### Client Validation

We discussed this with the client:

> **Us:** "The side-by-side chunk comparison we built matches by semantic similarity, not document structure. A 'MODIFIED' tag doesn't actually mean it's the same paragraph that changed — it means we found two chunks that sound 80% similar. Is that useful, or is it misleading?"

> **Client:** "Honestly? That's fine. The side-by-side was my mental model from doing this manually — two PDFs open, scrolling through both. But that's not actually what I *need*. What I need is to know what changed and where to look. If the system tells me 'Meta added new language about EU AI Act compliance in FY2024 that wasn't present in FY2023' and shows me the exact excerpt with a citation — that's the answer."

### The New Approach: Comparison-Aware Search

Instead of fake structural comparison, we implemented **Perplexity-style comparison**:

1. **Search both years** for the user's query
2. **Show hit counts** (e.g., "4 results in FY2024, 1 in FY2023") — this alone is a useful trend signal
3. **LLM compares excerpts directly** and identifies:
   - New topics in the newer year
   - Removed topics from the older year
   - **Wording changes** — including subtle shifts like "significant competition" → "intense and increasing competition"
4. **All sources cited** — analyst can verify any claim

### Why This Is Actually Better

| Old Approach (Removed) | New Approach |
|------------------------|--------------|
| Chunk similarity = structural match (false) | LLM compares meaning, not position |
| Could miss wording changes within same topic | Explicitly prompted to find wording shifts |
| "MODIFIED 78%" — what does that mean? | "Changed from X to Y" — clear explanation |
| Misleading confidence | Honest synthesis with citations |

### Client Confirmation

> "The Perplexity model makes sense. I use Perplexity. The citations are the key — it's not a black box."
>
> "For the 'MODIFIED' cases — where they didn't add something new, but changed the wording of an existing risk — can the system still surface that?"
>
> **Answer:** Yes — LLM is explicitly prompted to identify wording changes, tone shifts, and intensified language. This is actually where the new approach is *stronger* than structural diff.

### What Was Removed

- `compare_all_chunks()` — compared every chunk between years
- `compare_query_across_years()` — matched chunks by embedding similarity
- `render_compare_chunk()` — displayed MODIFIED/NEW/REMOVED tags
- `generate_word_diff()` / `render_side_by_side_diff()` — visual diff highlighting
- Tabs for New/Modified/Removed/Unchanged

### What Was Added

- `generate_comparison_summary()` — LLM prompt that explicitly compares excerpts from both years, identifies new/removed topics AND wording changes
- Hit counts by year in Search mode — trend visualization
- Simplified Compare UI — synthesis + sources, no misleading tags

### Final Architecture

```
Search Mode:
  Query → Search all years → Hit counts by year → LLM summary → Sources

Compare Mode:
  Query + Year A + Year B → Search both → Hit counts → LLM comparison → Sources
```

**Design pattern:** Perplexity-style — search + synthesize + cite sources. Honest about what we know and don't know.

---

## Lessons Learned

1. **RAG is great for retrieval, not for document structure** — Don't shoehorn chunk similarity into structural diff.
2. **Semantic similarity ≠ same thing** — 80% similar chunks might be completely different sections.
3. **False confidence is worse than no feature** — Better to be honest about limitations.
4. **LLM comparison > embedding comparison for meaning** — LLMs understand context, embeddings measure surface similarity.
5. **Client knows their workflow better than we do** — But they may describe it in terms of their current tools (two PDFs side-by-side). Dig into what they actually *need*.

---

## Unified UI Feedback (2025-12-28)

Client reviewed screenshots of new unified Search + Compare interface.

### What's Working Well

| Feature | Client Feedback |
|---------|-----------------|
| FY Trend Analysis | "Exactly what I asked for. That hit count by year is a quick signal before I even read the synthesis." |
| Trending Topics sidebar | "Smart. Those are literally the queries I said I'd run. Saves me typing." |
| Collapsible summaries | "Good. I asked for this. Sometimes I want the synthesis, sometimes I just want the raw excerpts." |

> "If I searched 'China' and saw FY2020: 2 → FY2024: 12, that tells me a story instantly."

### Requested Changes

| Issue | Request | Priority |
|-------|---------|----------|
| Compare mode synthesis | Should lead with "what's new this year vs last year" not just summarize the topic | High |
| Confidence metrics | "Where do those appear?" — need to be more prominent on each excerpt | High |
| Trending Topics | Should be clickable to run queries directly, not just suggestions | Medium |
| "Compliance Mode: Strict" | Confusing label — "What does that mean?" | Low (clarify or remove) |

### Export Requirements (Clarified)

- Excerpts clearly labeled with source (year, section)
- Must copy-paste cleanly into research template
- No weird formatting that breaks in Outlook

> "I haven't seen the actual Word export yet. Can you send me a sample?"

### Questions Answered

| Question | Answer |
|----------|--------|
| "Does Compare button switch to YoY mode?" | Yes — shows two year selectors, runs comparison |
| "What is Compliance Mode: Strict?" | Placeholder label — should clarify meaning or remove |
| "Can I click Trending Topics?" | Not yet — currently just text suggestions |

### Overall Assessment

> "This is usable. I want to run my test queries — Apple ATT, FTC antitrust, workforce reduction — and see real results. When can I get hands-on access?"

### Action Items Before Handoff

| Item | Priority | Complexity |
|------|----------|------------|
| Update Compare synthesis prompt to lead with "what changed" | P0 | Low |
| Make confidence indicators more prominent on excerpts | P0 | Low |
| Make Trending Topics clickable | P1 | Low |
| Clarify or remove "Compliance Mode: Strict" label | P2 | Trivial |
| Generate sample Word export for client review | P1 | None (just export) |
| Deploy for client testing | P0 | Low |

---

## Final Review (2025-12-27)

### Client Validation of Exports
Client reviewed `Meta_Risk_Report_Sample.pdf` and confirmed readiness for earnings season.

**What Works:**
- **Year breakdown trend** (e.g., "FY2020: 1 -> FY2024: 12") is copy-paste ready for notes.
- **AI Summary** is quotable for PMs.
- **Confidence scores** build trust.
- **Formatting** is clean (no broken tables in Outlook).

**Minor Feedback for V2 (Not Blockers):**
1. **Filing Date:** Add specific date (e.g., "Filed: Jan 2025") next to Fiscal Year to avoid ambiguity.
2. **Page Numbers:** Add citations (e.g., "p. 24") to allow cross-referencing original PDF.
3. **Export Filename:** Auto-name files descriptively, e.g., `META_AI_regulation_2024-12-27.pdf`.

### Go-No-Go Decision
**Status:** **GO** for Beta V1.
**Next Steps:** Keep test environment active through January earnings season.


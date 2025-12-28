"""
Meta 10-K Risk Factor Search - Comparison-Aware Search with LLM Synthesis
"""

import json
import pickle
from pathlib import Path
from collections import Counter

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Config
INDEX_DIR = "vector_store"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

# OpenRouter config
OPENROUTER_API_KEY = "REDACTED_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "google/gemini-2.0-flash-001"  # Fast and cheap for summaries

# Confidence thresholds (for search relevance)
HIGH_CONFIDENCE = 0.75      # > 0.75 = high confidence match
MEDIUM_CONFIDENCE = 0.55    # 0.55-0.75 = medium, < 0.55 = low


# Initialize OpenRouter client
@st.cache_resource
def get_llm_client():
    return OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)


def generate_search_summary(query: str, results: list) -> str:
    """Generate a brief summary of search results."""
    if not results:
        return None

    client = get_llm_client()

    # Format chunks for the prompt
    chunks_text = ""
    for i, r in enumerate(results[:10], 1):  # Limit to top 10 for prompt
        chunks_text += f"[{i}] FY{r['fiscal_year']}: {r['content'][:300]}...\n\n"

    prompt = f"""You are analyzing Meta's 10-K Risk Factor disclosures for an equity research analyst.

Query: "{query}"

Here are the top search results:

{chunks_text}

Provide a structured summary:

**Overview** (2-3 sentences): What does Meta disclose about "{query}"? Which fiscal years cover this?

**Key Excerpts:**
- [1]: Brief description of what this excerpt covers
- [2]: Brief description of what this excerpt covers
- [3]: Brief description of what this excerpt covers
(Continue for top 5-6 most relevant)

**Patterns/Trends** (1-2 sentences): Any notable evolution or patterns across years?

Keep the entire response to 1-2 short paragraphs. Be concise but complete."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Summary unavailable: {str(e)}"


def generate_comparison_summary(query: str, results_a: list, results_b: list, year_a: int, year_b: int) -> str:
    """
    Generate a comparison summary using LLM.

    The LLM compares excerpts from both years and identifies:
    - New topics in year_a
    - Removed topics from year_b
    - Wording changes (same topic, different language)
    """
    client = get_llm_client()

    # Format excerpts from newer year
    excerpts_a = ""
    for i, r in enumerate(results_a[:8], 1):
        excerpts_a += f"[A{i}] {r['content'][:400]}...\n\n"

    # Format excerpts from older year
    excerpts_b = ""
    for i, r in enumerate(results_b[:8], 1):
        excerpts_b += f"[B{i}] {r['content'][:400]}...\n\n"

    prompt = f"""You are comparing Meta's 10-K Risk Factor disclosures between FY{year_a} and FY{year_b} for an equity research analyst.

Query: "{query}"

**FY{year_a} excerpts ({len(results_a)} found):**
{excerpts_a if excerpts_a else "(No relevant excerpts found)"}

**FY{year_b} excerpts ({len(results_b)} found):**
{excerpts_b if excerpts_b else "(No relevant excerpts found)"}

Analyze these excerpts and identify:

1. **New in FY{year_a}**: Topics or concerns that appear in FY{year_a} but not FY{year_b}
2. **Removed from FY{year_b}**: Topics that were in FY{year_b} but not FY{year_a}
3. **Wording Changes**: Same topic but different language — flag intensification (e.g., "significant" → "intense and increasing"), softening, or shifted emphasis. Quote the specific wording from each year.
4. **Unchanged**: Topics covered similarly in both years (mention briefly)

For each finding, cite the specific excerpt numbers [A1], [B2], etc.

Format your response as:

**Summary** (2-3 sentences): What's the most important change related to "{query}"?

**Key Changes:**
- [NEW] Description... [cite A1, A2]
- [WORDING CHANGE] "old wording" → "new wording" [cite B1, A3]
- [REMOVED] Description... [cite B2]

**Interpretation** (1-2 sentences): What might these changes indicate for Meta?

Be specific about wording differences — analysts care about subtle language shifts. If hit counts differ significantly ({len(results_a)} vs {len(results_b)}), note whether this topic is getting more or less attention."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Summary unavailable: {str(e)}"


@st.cache_resource
def load_model_and_index():
    """Load the embedding model and FAISS index (cached)."""
    with open(f"{INDEX_DIR}/config.json") as f:
        config = json.load(f)

    model = SentenceTransformer(config["model_name"])
    index = faiss.read_index(f"{INDEX_DIR}/{INDEX_FILE}")

    with open(f"{INDEX_DIR}/{METADATA_FILE}", "rb") as f:
        metadata = pickle.load(f)

    return model, index, metadata


def search(query: str, model, index, metadata, top_k: int = 15, year_filter: int = None):
    """Standard search across all years."""
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding.astype(np.float32), k=top_k * 3)

    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx < 0:
            continue

        meta = metadata[idx]

        if year_filter and meta["fiscal_year"] != year_filter:
            continue

        results.append({
            "score": float(score),
            "chunk_id": meta["chunk_id"],
            "company": meta["company"],
            "fiscal_year": meta["fiscal_year"],
            "section": meta["section"],
            "chunk_type": meta["chunk_type"],
            "content": meta["content"],
            "parent_context": meta.get("parent_context", ""),
        })

        if len(results) >= top_k:
            break

    return results


def search_all_years(query: str, model, index, metadata, top_k_per_year: int = 10):
    """Search and return results grouped by year with counts."""
    query_embedding = model.encode([query], normalize_embeddings=True)
    # Get more results to ensure coverage across years
    distances, indices = index.search(query_embedding.astype(np.float32), k=top_k_per_year * 10)

    # Group by year
    results_by_year = {}
    for idx, score in zip(indices[0], distances[0]):
        if idx < 0:
            continue

        meta = metadata[idx]
        year = meta["fiscal_year"]

        if year not in results_by_year:
            results_by_year[year] = []

        if len(results_by_year[year]) < top_k_per_year:
            results_by_year[year].append({
                "score": float(score),
                "chunk_id": meta["chunk_id"],
                "company": meta["company"],
                "fiscal_year": meta["fiscal_year"],
                "section": meta["section"],
                "chunk_type": meta["chunk_type"],
                "content": meta["content"],
                "parent_context": meta.get("parent_context", ""),
            })

    return results_by_year


def get_hit_counts(results: list) -> dict:
    """Get hit counts by year from search results."""
    counts = Counter(r["fiscal_year"] for r in results)
    return dict(sorted(counts.items()))


def render_source_chunk(result: dict, ref_num: int, year_label: str = None):
    """Render a source chunk with citation number."""
    with st.container():
        col1, col2 = st.columns([0.5, 5.5])
        with col1:
            st.markdown(f"**[{ref_num}]**")
        with col2:
            year_str = f"FY{result['fiscal_year']}"
            if year_label:
                year_str = f"{year_label} ({year_str})"
            st.markdown(f"**{result['company']} {year_str}** · Score: {result['score']:.2f}")
            st.caption(result['section'])

        st.markdown(f"> {result['content'][:600]}{'...' if len(result['content']) > 600 else ''}")
        st.caption(f"ID: {result['chunk_id']}")
        st.divider()


# Page config
st.set_page_config(
    page_title="Meta 10-K Risk Factor Search",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("📊 Meta 10-K Risk Factor Search")
st.markdown("Search across Meta's Risk Factors (Item 1A) from FY2020-2024")

# Load resources
with st.spinner("Loading model and index..."):
    model, index, metadata = load_model_and_index()

# Get available years
years = sorted(set(m["fiscal_year"] for m in metadata))

# Sidebar - Mode selection
st.sidebar.header("Mode")
mode = st.sidebar.radio("Select Mode", ["🔍 Search", "📊 Compare Years"], index=0)

if mode == "🔍 Search":
    # --- SEARCH MODE ---
    st.sidebar.header("Filters")

    year_options = ["All Years"] + [f"FY{y}" for y in years]
    selected_year = st.sidebar.selectbox("Fiscal Year", year_options)

    year_filter = None
    if selected_year != "All Years":
        year_filter = int(selected_year.replace("FY", ""))

    top_k = st.sidebar.slider("Number of results", 5, 30, 15)

    # Search input
    query = st.text_input("🔍 Enter your search query", placeholder="e.g., AI regulation, Apple competition, China risks...")

    if query:
        results = search(query, model, index, metadata, top_k=top_k, year_filter=year_filter)

        if results:
            # Hit counts by year (trend visualization)
            hit_counts = get_hit_counts(results)

            st.markdown(f"### Found {len(results)} results")

            # Show hit counts as a trend indicator
            if len(hit_counts) > 1:
                st.markdown("**📈 Results by year:**")
                counts_str = " → ".join([f"FY{y}: **{c}**" for y, c in sorted(hit_counts.items())])
                st.markdown(counts_str)

            # Generate and display summary
            with st.spinner("Generating summary..."):
                summary = generate_search_summary(query, results)
            if summary:
                st.info(summary)

            st.markdown("---")
            st.markdown("### Sources")
            st.caption("**Score guide:** 0.7+ = strong match, 0.5-0.7 = moderate, <0.5 = weak")

            for i, r in enumerate(results, 1):
                render_source_chunk(r, i)
        else:
            st.warning("No results found. Try a different query or remove filters.")

else:
    # --- COMPARE MODE ---
    st.sidebar.header("Compare Settings")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        year_a = st.selectbox("Newer Year", [f"FY{y}" for y in reversed(years)], index=0)
    with col2:
        year_b = st.selectbox("Older Year", [f"FY{y}" for y in reversed(years)], index=1)

    year_a_int = int(year_a.replace("FY", ""))
    year_b_int = int(year_b.replace("FY", ""))

    top_k = st.sidebar.slider("Results per year", 5, 20, 10)

    # Query input
    st.markdown("### What changed?")
    query = st.text_input(
        "🔍 Enter topic to compare",
        placeholder="e.g., AI regulation, competition, antitrust...",
        key="compare_query"
    )

    if year_a_int == year_b_int:
        st.warning("Please select two different years to compare.")
    elif query:
        st.markdown(f"### {query}: {year_a} vs {year_b}")

        # Search both years
        with st.spinner("Searching both years..."):
            results_a = search(query, model, index, metadata, top_k=top_k, year_filter=year_a_int)
            results_b = search(query, model, index, metadata, top_k=top_k, year_filter=year_b_int)

        # Show hit counts
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"📊 {year_a}", f"{len(results_a)} results")
        with col2:
            st.metric(f"📊 {year_b}", f"{len(results_b)} results")

        # Trend indicator
        if len(results_a) > len(results_b):
            st.success(f"↑ More coverage in {year_a} — topic appears to have increased emphasis")
        elif len(results_b) > len(results_a):
            st.warning(f"↓ Less coverage in {year_a} — topic appears to have decreased emphasis")
        else:
            st.info("≈ Similar coverage in both years")

        # Generate comparison summary
        if results_a or results_b:
            with st.spinner("Analyzing differences..."):
                summary = generate_comparison_summary(query, results_a, results_b, year_a_int, year_b_int)

            st.markdown("---")
            st.markdown("### Analysis")
            st.info(summary)

            st.markdown("---")
            st.markdown("### Sources")

            # Show sources from newer year first
            if results_a:
                st.markdown(f"**{year_a} (Newer)**")
                for i, r in enumerate(results_a, 1):
                    render_source_chunk(r, f"A{i}", "Newer")

            if results_b:
                st.markdown(f"**{year_b} (Older)**")
                for i, r in enumerate(results_b, 1):
                    render_source_chunk(r, f"B{i}", "Older")
        else:
            st.warning(f"No results found for '{query}' in either year. Try different keywords.")
    else:
        st.info("Enter a topic above to see what changed between the selected years.")

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("### Example Queries")
    if mode == "🔍 Search":
        st.markdown("""
        - AI regulation
        - Apple iOS ATT
        - China risks
        - FTC antitrust
        - workforce reduction
        - metaverse
        """)
    else:
        st.markdown("""
        **Try comparing:**
        - "AI regulation"
        - "competition"
        - "antitrust"
        - "workforce reduction"
        - "China"
        """)

    st.markdown("---")
    st.markdown("### How It Works")
    if mode == "🔍 Search":
        st.markdown("""
        1. Search finds relevant excerpts
        2. Shows hit counts by year (trend signal)
        3. LLM synthesizes findings
        4. Sources provided for verification
        """)
    else:
        st.markdown("""
        1. Search both years for your query
        2. LLM compares excerpts directly
        3. Identifies new, removed, and wording changes
        4. All sources cited for verification
        """)

    st.markdown("---")
    st.markdown("### Index Stats")
    st.markdown(f"- **Documents:** {index.ntotal:,}")
    st.markdown(f"- **Years:** FY{min(years)}-FY{max(years)}")

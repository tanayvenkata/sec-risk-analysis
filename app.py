"""
Meta 10-K Risk Factor Search - Comparison-Aware Search with LLM Synthesis
"""

import json
import os
import pickle
from dotenv import load_dotenv

load_dotenv()
from pathlib import Path
from collections import Counter

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from docx import Document
from docx.shared import Pt
from fpdf import FPDF
import io

# Config
INDEX_DIR = "vector_store"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

# OpenRouter config
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY environment variable is required. Set it in your .env file.")
    st.stop()
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

**What Changed** (lead with this — 2-3 sentences): Start with the most significant difference between FY{year_a} and FY{year_b} regarding "{query}". Be direct: "Meta added...", "Meta removed...", "Meta intensified language about..."

**Key Changes:**
- [NEW] What's in FY{year_a} that wasn't in FY{year_b}... [cite A1]
- [WORDING] "old phrase" → "new phrase" — note if language intensified or softened [cite B1, A3]
- [REMOVED] What was in FY{year_b} but not FY{year_a}... [cite B2]

**Bottom Line** (1 sentence): What should an analyst take away from these changes?

Be specific about wording differences. Quote exact phrases when noting changes."""

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


def search(query: str, model, index, metadata, top_k: int = 15, year_filter: int = None, company_filter: list = None, section_filter: list = None):
    """Standard search with optional filters."""
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding.astype(np.float32), k=top_k * 5)

    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx < 0:
            continue

        meta = metadata[idx]

        # Apply filters
        if year_filter and meta["fiscal_year"] != year_filter:
            continue
        if company_filter and meta["company"] not in company_filter:
            continue
        if section_filter:
            section_match = any(f in meta["section"] for f in section_filter)
            if not section_match:
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


def export_to_word(query: str, summary: str, results: list, mode: str, metadata: dict = None) -> bytes:
    """Export results to Word document."""
    doc = Document()

    # Title
    doc.add_heading(f"Meta 10-K Risk Factor {mode}", 0)

    # Query
    doc.add_heading("Query", level=1)
    doc.add_paragraph(query)

    # Metadata (hit counts, years compared, etc.)
    if metadata:
        doc.add_heading("Overview", level=1)
        for key, value in metadata.items():
            doc.add_paragraph(f"{key}: {value}")

    # Summary
    if summary:
        doc.add_heading("AI Summary", level=1)
        doc.add_paragraph(summary)

    # Sources
    doc.add_heading("Sources", level=1)
    for i, r in enumerate(results, 1):
        ref = r.get('ref', str(i))
        p = doc.add_paragraph()
        run = p.add_run(f"[{ref}] {r['company']} FY{r['fiscal_year']}")
        run.bold = True
        p.add_run(f" · Score: {r['score']:.2f}")
        doc.add_paragraph(r['section'], style='Caption')
        content = r['content'][:600] + ('...' if len(r['content']) > 600 else '')
        doc.add_paragraph(content)
        doc.add_paragraph()  # spacer

    # Save to bytes
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def export_to_pdf(query: str, summary: str, results: list, mode: str, metadata: dict = None) -> bytes:
    """Export results to PDF document."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Meta 10-K Risk Factor {mode}", ln=True)
    pdf.ln(5)

    # Query
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Query:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, query)
    pdf.ln(3)

    # Metadata
    if metadata:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Overview:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for key, value in metadata.items():
            pdf.cell(0, 6, f"{key}: {value}", ln=True)
        pdf.ln(3)

    # Summary
    if summary:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "AI Summary:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        # Clean summary for PDF (remove markdown)
        clean_summary = summary.replace('**', '').replace('*', '').replace('#', '')
        pdf.multi_cell(0, 5, clean_summary)
        pdf.ln(3)

    # Sources
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Sources:", ln=True)
    pdf.ln(2)
    for i, r in enumerate(results, 1):
        ref = r.get('ref', str(i))
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, f"[{ref}] {r['company']} FY{r['fiscal_year']} - Score: {r['score']:.2f}", ln=True)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 5, r['section'][:80], ln=True)
        pdf.set_font("Helvetica", "", 9)
        content = r['content'][:500] + ('...' if len(r['content']) > 500 else '')
        pdf.multi_cell(0, 4, content)
        pdf.ln(2)

    return bytes(pdf.output())


def render_collapsible_summary(summary: str, title: str = "Summary", expanded: bool = False, allow_html: bool = False):
    """Render a collapsible summary with preview."""
    if not summary:
        return

    # Extract first sentence or first 150 chars as preview
    preview = summary.split('\n')[0][:150]
    if len(summary) > 150:
        preview += "..."

    label = f"{title} (Collapse)" if expanded else f"{title} (Expand)"
    with st.expander(label, expanded=expanded):
        st.markdown(summary, unsafe_allow_html=allow_html)

    # Show brief preview outside expander
    if not expanded:
        st.caption(f"*{preview}*")


def get_confidence_meta(score):
    """Return label, css class, and shape class based on score."""
    if score >= 0.75:
        return "High Relevance", "confidence-high", "shape-square"
    elif score >= 0.55:
        return "Medium Relevance", "confidence-med", "shape-triangle"
    else:
        return "Low Relevance", "confidence-low", "shape-circle"


def render_source_chunk(result: dict, ref_num: int, year_label: str = None):
    """Render a source chunk using the accessible Card component."""
    
    label, confidence_class, shape_class = get_confidence_meta(result['score'])
    
    year_display = f"FY{result['fiscal_year']}"
    if year_label:
        year_display = f"{year_label} ({year_display})"

    # Escape HTML content to prevent rendering issues if content has tags
    import html as html_lib
    safe_content = html_lib.escape(result['content'][:600])
    if len(result['content']) > 600:
        safe_content += "..."

    html = f"""
    <div class="result-card">
        <div class="card-header">
            <div class="card-meta">
                <span class="fiscal-year-badge">{year_display}</span>
                <span class="section-label">{result['section']}</span>
            </div>
            <div class="confidence-indicator {confidence_class} {shape_class}" 
                 title="Score: {result['score']:.2f}"
                 aria-label="{label}, Score: {result['score']:.2f}">
                {label}
            </div>
        </div>
        <div class="excerpt-text">
            {safe_content}
        </div>
        <div style="margin-top: 1rem; font-size: 0.8rem; color: #666; display: flex; justify-content: space-between;">
            <span><strong>Ref [{ref_num}]</strong> • ID: {result['chunk_id']}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# Page config
st.set_page_config(
    page_title="Meta 10-K Analysis Platform",
    page_icon=None,
    layout="wide"
)

def load_css():
    with open("assets/custom.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Header
st.title("10-K Risk Analysis Platform")

# Load resources first (needed for dynamic header)
with st.spinner("Loading model and index..."):
    model, index, metadata = load_model_and_index()

# Get available years and companies
years = sorted(set(m["fiscal_year"] for m in metadata))
all_companies = sorted(set(m["company"] for m in metadata))

# Format companies for display
if len(all_companies) <= 2:
    companies_display = " & ".join(all_companies)
else:
    companies_display = f"{len(all_companies)} Companies"

# Institutional Dashboard Header (now with dynamic values)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-label">Coverage</div>
        <div class="metric-value">Risk Factors + MD&A</div>
        <div style="font-size: 0.7rem; color: #888; margin-top: 4px;">Item 1A & Item 7</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    year_range = f"{min(years)}-{max(years)}" if years else "N/A"
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Fiscal Years</div>
        <div class="metric-value">{year_range}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Companies</div>
        <div class="metric-value">{companies_display}</div>
        <div style="font-size: 0.7rem; color: #888; margin-top: 4px;">{', '.join(all_companies)}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Session state for compare mode and query
if "compare_mode" not in st.session_state:
    st.session_state.compare_mode = False
if "selected_query" not in st.session_state:
    st.session_state.selected_query = ""

# Sidebar header
st.sidebar.header("10-K Risk Analytics")

# Derive section types from metadata (sections starting with "MDA:" are MD&A, others are Risk Factors)
section_types_found = set()
for m in metadata:
    if m["section"].startswith("MDA:"):
        section_types_found.add("MDA")
    else:
        section_types_found.add("Risk Factors")
all_sections = sorted(section_types_found)

# Company filter with "All Companies" toggle
select_all_companies = st.sidebar.checkbox("All Companies (sector view)", value=False)
if select_all_companies:
    selected_companies = all_companies
    st.sidebar.multiselect(
        "Companies",
        options=all_companies,
        default=all_companies,
        disabled=True
    )
else:
    selected_companies = st.sidebar.multiselect(
        "Companies",
        options=all_companies,
        default=[all_companies[0]] if all_companies else []
    )

# Section filter
selected_sections = st.sidebar.multiselect(
    "Sections",
    options=all_sections,
    default=["Risk Factors"]
)

st.sidebar.markdown("---")

# Unified Filter Toolbar
year_options = [f"FY{y}" for y in reversed(years)]

if st.session_state.compare_mode:
    # Compare mode: two year selectors
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        year_a = st.selectbox("Newer Year", year_options, index=0)
    with col2:
        year_b = st.selectbox("Older Year", year_options, index=1)
    with col3:
        top_k = st.slider("Result Limit", 5, 20, 10)
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Cancel", type="secondary"):
            st.session_state.compare_mode = False
            st.rerun()

    year_a_int = int(year_a.replace("FY", ""))
    year_b_int = int(year_b.replace("FY", ""))
    year_filter = None  # Not used in compare mode
else:
    # Search mode: single year + compare button
    col1, col2, col3 = st.columns([2, 2, 4])
    with col1:
        all_year_options = ["All Years"] + year_options
        selected_year = st.selectbox("Fiscal Year Scope", all_year_options)
        year_filter = None
        if selected_year != "All Years":
            year_filter = int(selected_year.replace("FY", ""))
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("+ Compare", type="primary"):
            st.session_state.compare_mode = True
            st.rerun()
    with col3:
        top_k = st.slider("Result Limit", 5, 30, 15)

# Query input
query_label = "Compare Topic" if st.session_state.compare_mode else "Risk Query"
query_placeholder = "e.g., AI regulation, competition, antitrust..." if st.session_state.compare_mode else "e.g., AI regulation, Apple competition, China risks..."
query = st.text_input(query_label, value=st.session_state.selected_query, placeholder=query_placeholder)
# Clear selected_query after it's been used
if st.session_state.selected_query:
    st.session_state.selected_query = ""

# Results section - adapts based on mode
if st.session_state.compare_mode:
    # --- COMPARE MODE ---
    if year_a_int == year_b_int:
        st.warning("Please select two different years to compare.")
    elif query:
        st.markdown(f"### {query}: {year_a} vs {year_b}")

        # Search both years
        with st.spinner("Searching both years..."):
            results_a = search(query, model, index, metadata, top_k=top_k, year_filter=year_a_int, company_filter=selected_companies, section_filter=selected_sections)
            results_b = search(query, model, index, metadata, top_k=top_k, year_filter=year_b_int, company_filter=selected_companies, section_filter=selected_sections)

        # Show hit counts
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{year_a}", f"{len(results_a)} results")
        with col2:
            st.metric(f"{year_b}", f"{len(results_b)} results")

        # Trend indicator
        if len(results_a) > len(results_b):
            st.success(f"Increased Focus in {year_a} (Coverage expanded)")
        elif len(results_b) > len(results_a):
            st.warning(f"Decreased Focus in {year_a} (Coverage reduced)")
        else:
            st.info("Stable Coverage")

        # Generate comparison summary
        if results_a or results_b:
            with st.spinner("Analyzing differences..."):
                summary = generate_comparison_summary(query, results_a, results_b, year_a_int, year_b_int)

                # Post-process summary for accessible styling
                if summary:
                    summary = summary.replace("[NEW]", '')
                    summary = summary.replace("[REMOVED]", '')
                    summary = summary.replace("[IMPORTANT]", '<span class="diff-added">')
                    summary = summary.replace("[REMOVED]", '<span class="diff-removed">')
                    summary = summary.replace("[/IMPORTANT]", '</span>')
                    summary = summary.replace("[/REMOVED]", '</span>')

            st.markdown("---")
            st.markdown("### Comparison Matrix")
            render_collapsible_summary(summary, title="Change Analysis", expanded=True, allow_html=True)

            st.markdown("---")
            st.markdown("### Sources")

            if results_a:
                st.markdown(f"**{year_a} (Newer)**")
                for i, r in enumerate(results_a, 1):
                    render_source_chunk(r, f"A{i}", "Newer")

            if results_b:
                st.markdown(f"**{year_b} (Older)**")
                for i, r in enumerate(results_b, 1):
                    render_source_chunk(r, f"B{i}", "Older")

            # Export buttons
            st.markdown("---")
            st.markdown("### Export")
            exp_col1, exp_col2 = st.columns(2)

            all_results = [{"ref": f"A{i+1}", **r} for i, r in enumerate(results_a)] + \
                          [{"ref": f"B{i+1}", **r} for i, r in enumerate(results_b)]

            compare_metadata = {
                "Comparison": f"{year_a} vs {year_b}",
                f"{year_a} Results": len(results_a),
                f"{year_b} Results": len(results_b),
            }

            with exp_col1:
                word_bytes = export_to_word(query, summary, all_results, "Comparison", compare_metadata)
                st.download_button(
                    "Download Word",
                    data=word_bytes,
                    file_name=f"meta_compare_{query[:20].replace(' ', '_')}_{year_a}_vs_{year_b}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

            with exp_col2:
                pdf_bytes = export_to_pdf(query, summary, all_results, "Comparison", compare_metadata)
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=f"meta_compare_{query[:20].replace(' ', '_')}_{year_a}_vs_{year_b}.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("No results found. Try a different query.")
    else:
        st.info("Enter a topic above to see what changed between the selected years.")

else:
    # --- SEARCH MODE ---
    if query:
        results = search(query, model, index, metadata, top_k=top_k, year_filter=year_filter, company_filter=selected_companies, section_filter=selected_sections)

        if results:
            hit_counts = get_hit_counts(results)

            st.markdown(f"### Found {len(results)} results")

            if len(hit_counts) > 1:
                st.markdown("**FY Trend Analysis:**")
                counts_str = " → ".join([f"FY{y}: **{c}**" for y, c in sorted(hit_counts.items())])
                st.markdown(counts_str)

            with st.spinner("Synthesizing analysis..."):
                summary = generate_search_summary(query, results)
            if summary:
                render_collapsible_summary(summary, title="Executive Synthesis", expanded=True)

            st.markdown("---")
            st.markdown("### Sources")
            st.caption("**Score guide:** 0.7+ = strong match, 0.5-0.7 = moderate, <0.5 = weak")

            for i, r in enumerate(results, 1):
                render_source_chunk(r, i)

            # Export buttons
            st.markdown("---")
            st.markdown("### Export")
            exp_col1, exp_col2 = st.columns(2)

            export_metadata = {
                "Total Results": len(results),
                "Years": ", ".join([f"FY{y}: {c}" for y, c in hit_counts.items()])
            }

            with exp_col1:
                word_bytes = export_to_word(query, summary, results, "Search", export_metadata)
                st.download_button(
                    "Download Word",
                    data=word_bytes,
                    file_name=f"meta_search_{query[:20].replace(' ', '_')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

            with exp_col2:
                pdf_bytes = export_to_pdf(query, summary, results, "Search", export_metadata)
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=f"meta_search_{query[:20].replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("No results found. Try a different query or remove filters.")

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("### Trending Topics")
    st.caption("Click to search")

    if st.session_state.compare_mode:
        topics = ["AI regulation", "competition", "antitrust", "workforce reduction", "China"]
    else:
        topics = ["AI regulation", "Apple iOS ATT", "China risks", "FTC antitrust", "workforce reduction", "metaverse"]

    for topic in topics:
        if st.button(topic, key=f"topic_{topic}", use_container_width=True):
            st.session_state.selected_query = topic
            st.rerun()

    st.markdown("---")
    st.markdown("### System Methodology")
    if st.session_state.compare_mode:
        st.markdown("""
        1. Search both years for your query
        2. LLM compares excerpts directly
        3. Identifies new, removed, and wording changes
        4. All sources cited for verification
        """)
    else:
        st.markdown("""
        1. Search finds relevant excerpts
        2. Shows hit counts by year (trend signal)
        3. LLM synthesizes findings
        4. Sources provided for verification
        """)

    st.markdown("---")
    st.markdown("### Index Stats")
    st.markdown(f"- **Documents:** {index.ntotal:,}")
    st.markdown(f"- **Years:** FY{min(years)}-FY{max(years)}")

#!/usr/bin/env python3
"""
Chunk cleaned 10-K sections (Risk Factors and MD&A) for RAG indexing.

Strategy (Hybrid - preserve structure, enforce size limits):
- Section headers → metadata only (not chunked)
- Lead sentences → own chunk (high retrieval value)
- Bullet groups → single chunk, split if >512 tokens
- Paragraphs → single chunk, split if >512 tokens

Metadata stored per chunk:
- chunk_id, company, fiscal_year, filed_at, section, chunk_type, content, token_count

Embedding prefix (added in embed_and_index.py):
- [Company] [FY{year}] [Section] prepended to content for semantic matching
"""

import json
import re
from pathlib import Path

BASE_DIR = "sec_corpus"
OUTPUT_DIR = "sec_corpus/_chunks"  # Unified output for all companies

# Section headers to detect - Risk Factors
RISK_FACTOR_HEADERS = [
    "Item 1A. Risk Factors",
    "Summary Risk Factors",
    "Risks Related to Our Product Offerings",
    "Risks Related to Our Business Operations and Financial Results",
    "Risks Related to Business Operations and Financial Results",
    "Risks Related to Government Regulation and Enforcement",
    "Risks Related to Data, Security, Platform Integrity, and Intellectual Property",
    "Risks Related to Data, Security, and Intellectual Property",
    "Risks Related to Ownership of Our Class A Common Stock",
]

# Section headers to detect - MD&A (Item 7)
MDA_HEADERS = [
    "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "Item 7. Management's Discussion and Analysis",
    "Overview",
    "Results of Operations",
    "Liquidity and Capital Resources",
    "Critical Accounting Policies and Estimates",
    "Critical Accounting Estimates",
    "Recent Accounting Pronouncements",
    "Contractual Obligations",
]

# Combined headers
ALL_SECTION_HEADERS = RISK_FACTOR_HEADERS + MDA_HEADERS

# Approximate tokens (words * 1.3)
MAX_CHUNK_TOKENS = 512
OVERLAP_TOKENS = 50


def count_tokens(text: str) -> int:
    """Rough token count (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def is_section_header(line: str) -> bool:
    """Check if line is a section header."""
    return line.strip() in ALL_SECTION_HEADERS


def is_bullet(line: str) -> bool:
    """Check if line is a bullet point."""
    return line.strip().startswith("•")


def is_lead_sentence(line: str) -> bool:
    """
    Detect lead sentences - they tend to be:
    - Not bullets
    - End with period
    - Relatively short (< 200 chars usually)
    - Often contain "may", "could", "if", strong language
    """
    line = line.strip()
    if is_bullet(line) or is_section_header(line):
        return False
    # Lead sentences are typically shorter thesis statements
    if len(line) < 300 and line.endswith('.'):
        # Common patterns in lead sentences
        lead_patterns = [
            r'\bmay\b.*\bharm',
            r'\bcould\b.*\baffect',
            r'\bfail to\b',
            r'\bdepend[s]?\b',
            r'\bsubject to\b',
            r'\bgenerate\b.*\brevenue\b',
            r'\bsignificantly\b',
            r'\bseriously\b',
            r'\bmaterially\b',
        ]
        for pattern in lead_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
    return False


def split_long_text(text: str, max_tokens: int = MAX_CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    """Split long text into chunks with overlap."""
    if count_tokens(text) <= max_tokens:
        return [text]

    words = text.split()
    chunks = []
    # Convert tokens to approximate word count
    max_words = int(max_tokens / 1.3)
    overlap_words = int(overlap / 1.3)

    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap_words
        if start >= len(words) - overlap_words:
            break

    return chunks


def split_bullet_group(bullets: list[str], max_tokens: int = MAX_CHUNK_TOKENS) -> list[str]:
    """Split a bullet group into chunks, keeping bullets intact."""
    if count_tokens("\n".join(bullets)) <= max_tokens:
        return ["\n".join(bullets)]

    chunks = []
    current_bullets = []
    current_tokens = 0

    for bullet in bullets:
        bullet_tokens = count_tokens(bullet)

        # If adding this bullet exceeds limit, save current chunk and start new
        if current_tokens + bullet_tokens > max_tokens and current_bullets:
            chunks.append("\n".join(current_bullets))
            current_bullets = []
            current_tokens = 0

        current_bullets.append(bullet)
        current_tokens += bullet_tokens

    # Don't forget the last chunk
    if current_bullets:
        chunks.append("\n".join(current_bullets))

    return chunks


def parse_document(lines: list[str]) -> list[dict]:
    """
    Parse document into structured blocks.

    Returns list of:
    - {"type": "section_header", "content": "..."}
    - {"type": "lead_sentence", "content": "..."}
    - {"type": "paragraph", "content": "..."}
    - {"type": "bullet_group", "content": "...", "bullets": [...]}
    """
    blocks = []
    current_bullets = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            # If we have accumulated bullets, save them
            if current_bullets:
                blocks.append({
                    "type": "bullet_group",
                    "content": "\n".join(current_bullets),
                    "bullets": current_bullets.copy()
                })
                current_bullets = []
            i += 1
            continue

        # Section header
        if is_section_header(line):
            if current_bullets:
                blocks.append({
                    "type": "bullet_group",
                    "content": "\n".join(current_bullets),
                    "bullets": current_bullets.copy()
                })
                current_bullets = []
            blocks.append({"type": "section_header", "content": line})
            i += 1
            continue

        # Bullet point
        if is_bullet(line):
            current_bullets.append(line)
            i += 1
            continue

        # Non-bullet line - save any accumulated bullets first
        if current_bullets:
            blocks.append({
                "type": "bullet_group",
                "content": "\n".join(current_bullets),
                "bullets": current_bullets.copy()
            })
            current_bullets = []

        # Check if lead sentence
        if is_lead_sentence(line):
            blocks.append({"type": "lead_sentence", "content": line})
        else:
            blocks.append({"type": "paragraph", "content": line})

        i += 1

    # Don't forget trailing bullets
    if current_bullets:
        blocks.append({
            "type": "bullet_group",
            "content": "\n".join(current_bullets),
            "bullets": current_bullets.copy()
        })

    return blocks


def blocks_to_chunks(blocks: list[dict], company: str, fiscal_year: int, filed_at: str, section_type: str) -> list[dict]:
    """
    Convert parsed blocks into chunks with metadata.

    section_type: "Risk Factors" or "MDA" - used as prefix in section field
    """
    chunks = []
    # Default section depends on document type
    if section_type == "MDA":
        current_section = "MDA: Overview"
    else:
        current_section = "Risk Factors: Introduction"

    chunk_index = 0

    for block in blocks:
        # Update section tracking
        if block["type"] == "section_header":
            header = block["content"]
            # Prefix with section type for clarity
            if section_type == "MDA":
                current_section = f"MDA: {header}"
            else:
                current_section = header  # Risk factor headers are already descriptive
            continue

        # Create chunk based on block type
        if block["type"] == "lead_sentence":
            chunks.append({
                "chunk_id": f"{company}_FY{fiscal_year}_{section_type}_{chunk_index:04d}",
                "company": company,
                "fiscal_year": fiscal_year,
                "filed_at": filed_at,
                "section": current_section,
                "chunk_type": "lead_sentence",
                "content": block["content"],
                "token_count": count_tokens(block["content"]),
            })
            chunk_index += 1

        elif block["type"] == "bullet_group":
            # Split large bullet groups while keeping bullets intact
            bullet_chunks = split_bullet_group(block["bullets"])
            for j, bullet_chunk in enumerate(bullet_chunks):
                chunks.append({
                    "chunk_id": f"{company}_FY{fiscal_year}_{section_type}_{chunk_index:04d}",
                    "company": company,
                    "fiscal_year": fiscal_year,
                    "filed_at": filed_at,
                    "section": current_section,
                    "chunk_type": "bullet_group" if len(bullet_chunks) == 1 else f"bullet_group_part_{j+1}_of_{len(bullet_chunks)}",
                    "content": bullet_chunk,
                    "token_count": count_tokens(bullet_chunk),
                })
                chunk_index += 1

        elif block["type"] == "paragraph":
            # Split long paragraphs
            para_chunks = split_long_text(block["content"])
            for j, para_chunk in enumerate(para_chunks):
                chunks.append({
                    "chunk_id": f"{company}_FY{fiscal_year}_{section_type}_{chunk_index:04d}",
                    "company": company,
                    "fiscal_year": fiscal_year,
                    "filed_at": filed_at,
                    "section": current_section,
                    "chunk_type": "paragraph" if len(para_chunks) == 1 else f"paragraph_part_{j+1}_of_{len(para_chunks)}",
                    "content": para_chunk,
                    "token_count": count_tokens(para_chunk),
                })
                chunk_index += 1

    return chunks


def detect_section_type(filename: str) -> str:
    """Detect section type from filename."""
    if "MDA" in filename or "mda" in filename.lower():
        return "MDA"
    return "Risk Factors"


def process_company(company_dir: Path) -> list[dict]:
    """Process all cleaned files for a single company."""
    cleaned_dir = company_dir / "cleaned"
    if not cleaned_dir.exists():
        print(f"  No cleaned directory found in {company_dir}")
        return []

    # Load metadata
    metadata_file = company_dir / "metadata.json"
    if not metadata_file.exists():
        print(f"  No metadata.json found in {company_dir}")
        return []

    with open(metadata_file) as f:
        metadata = json.load(f)

    all_chunks = []

    # Process each cleaned file
    for txt_file in sorted(cleaned_dir.glob("FY*.txt")):
        print(f"  Processing {txt_file.name}...")

        # Extract fiscal year from filename
        year_match = re.search(r'FY(\d{4})', txt_file.name)
        if not year_match:
            print(f"    Warning: Could not extract year from {txt_file.name}")
            continue

        fiscal_year = int(year_match.group(1))
        section_type = detect_section_type(txt_file.name)

        # Find metadata for this file
        file_meta = None
        for m in metadata.get("filings", []):
            if m["fiscal_year"] == fiscal_year:
                # Match by section if available
                meta_section = m.get("section", "Risk Factors")
                if section_type == "MDA" and "MDA" in meta_section:
                    file_meta = m
                    break
                elif section_type == "Risk Factors" and "Risk" in meta_section:
                    file_meta = m
                    break
                elif "section" not in m:
                    # Legacy metadata without section field
                    file_meta = m
                    break

        filed_at = file_meta["filed_at"] if file_meta else "unknown"
        company = file_meta.get("ticker", company_dir.name) if file_meta else company_dir.name

        # Read file
        text = txt_file.read_text(encoding="utf-8")
        lines = text.split("\n")

        # Parse into blocks
        blocks = parse_document(lines)

        # Convert to chunks
        chunks = blocks_to_chunks(blocks, company, fiscal_year, filed_at, section_type)
        all_chunks.extend(chunks)

        print(f"    → {len(chunks)} chunks ({section_type})")

    return all_chunks


def main():
    base_path = Path(BASE_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    # Iterate over company directories
    for company_dir in sorted(base_path.iterdir()):
        # Skip non-directories and special directories
        if not company_dir.is_dir() or company_dir.name.startswith("_"):
            continue

        print(f"\nProcessing {company_dir.name}...")
        chunks = process_company(company_dir)
        all_chunks.extend(chunks)

    # Save combined chunks
    combined_output = output_path / "all_chunks.json"
    with open(combined_output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    # Summary
    print("\n" + "="*60)
    print("CHUNKING SUMMARY")
    print("="*60)
    print(f"Total chunks: {len(all_chunks)}")

    # By company
    companies = {}
    for c in all_chunks:
        companies[c["company"]] = companies.get(c["company"], 0) + 1

    print(f"\nBy company:")
    for company, count in sorted(companies.items()):
        print(f"  {company}: {count}")

    # By type
    types = {}
    for c in all_chunks:
        t = c["chunk_type"]
        if t.startswith("paragraph_part"):
            t = "paragraph_split"
        if t.startswith("bullet_group_part"):
            t = "bullet_group_split"
        types[t] = types.get(t, 0) + 1

    print(f"\nBy chunk type:")
    for t, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    # By section type (Risk Factors vs MDA)
    section_types = {"Risk Factors": 0, "MDA": 0}
    for c in all_chunks:
        if c["section"].startswith("MDA"):
            section_types["MDA"] += 1
        else:
            section_types["Risk Factors"] += 1

    print(f"\nBy section type:")
    for st, count in section_types.items():
        print(f"  {st}: {count}")

    # Token stats
    if all_chunks:
        tokens = [c["token_count"] for c in all_chunks]
        print(f"\nToken counts:")
        print(f"  Min: {min(tokens)}")
        print(f"  Max: {max(tokens)}")
        print(f"  Avg: {sum(tokens)/len(tokens):.0f}")

    print(f"\nOutput saved to: {combined_output}")


if __name__ == "__main__":
    main()

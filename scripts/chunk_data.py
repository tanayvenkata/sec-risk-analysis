#!/usr/bin/env python3
"""
Chunk cleaned Meta 10-K Risk Factors for RAG indexing.

Strategy (Hybrid - preserve structure, enforce size limits):
- Section headers → metadata only (not chunked)
- Lead sentences → own chunk (high retrieval value)
- Bullet groups → single chunk, split if >512 tokens
- Paragraphs → single chunk, split if >512 tokens

Metadata stored per chunk:
- chunk_id, company, fiscal_year, filed_at, section, chunk_type, content, token_count

Embedding prefix (added in embed_and_index.py):
- [FY{year}] [Section] prepended to content for semantic matching
"""

import json
import re
from pathlib import Path

INPUT_DIR = "sec_corpus/META/cleaned"
OUTPUT_DIR = "sec_corpus/META/chunked"

# Section headers to detect
SECTION_HEADERS = [
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

# Approximate tokens (words * 1.3)
MAX_CHUNK_TOKENS = 512
OVERLAP_TOKENS = 50


def count_tokens(text: str) -> int:
    """Rough token count (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def is_section_header(line: str) -> bool:
    """Check if line is a section header."""
    return line.strip() in SECTION_HEADERS


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


def blocks_to_chunks(blocks: list[dict], company: str, fiscal_year: int, filed_at: str) -> list[dict]:
    """Convert parsed blocks into chunks with metadata."""
    chunks = []
    current_section = "Introduction"
    chunk_index = 0

    for block in blocks:
        # Update section tracking
        if block["type"] == "section_header":
            current_section = block["content"]
            continue

        # Create chunk based on block type
        if block["type"] == "lead_sentence":
            chunks.append({
                "chunk_id": f"{company}_FY{fiscal_year}_{chunk_index:04d}",
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
                    "chunk_id": f"{company}_FY{fiscal_year}_{chunk_index:04d}",
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
                    "chunk_id": f"{company}_FY{fiscal_year}_{chunk_index:04d}",
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


def process_file(filepath: Path, metadata: dict) -> list[dict]:
    """Process a single cleaned risk factors file."""
    # Extract fiscal year from filename
    year_match = re.search(r'FY(\d{4})', filepath.name)
    if not year_match:
        print(f"  Warning: Could not extract year from {filepath.name}")
        return []

    fiscal_year = int(year_match.group(1))

    # Find metadata for this file
    file_meta = None
    for m in metadata["filings"]:
        if m["fiscal_year"] == fiscal_year:
            file_meta = m
            break

    filed_at = file_meta["filed_at"] if file_meta else "unknown"
    company = file_meta["ticker"] if file_meta else "META"

    # Read file
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")

    # Parse into blocks
    blocks = parse_document(lines)

    # Convert to chunks
    chunks = blocks_to_chunks(blocks, company, fiscal_year, filed_at)

    return chunks


def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_file = Path("sec_corpus/META/metadata.json")
    with open(metadata_file) as f:
        metadata = json.load(f)

    all_chunks = []

    # Process each file
    for txt_file in sorted(input_path.glob("FY*.txt")):
        print(f"Processing {txt_file.name}...")
        chunks = process_file(txt_file, metadata)
        all_chunks.extend(chunks)

        # Save per-year chunks
        year_output = output_path / f"{txt_file.stem}_chunks.json"
        with open(year_output, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        print(f"  → {len(chunks)} chunks")

    # Save combined chunks
    combined_output = output_path / "all_chunks.json"
    with open(combined_output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    # Summary
    print("\n" + "="*60)
    print("CHUNKING SUMMARY")
    print("="*60)
    print(f"Total chunks: {len(all_chunks)}")

    # By type
    types = {}
    for c in all_chunks:
        t = c["chunk_type"]
        if t.startswith("paragraph_part"):
            t = "paragraph_split"
        types[t] = types.get(t, 0) + 1

    print(f"\nBy chunk type:")
    for t, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    # By year
    years = {}
    for c in all_chunks:
        years[c["fiscal_year"]] = years.get(c["fiscal_year"], 0) + 1

    print(f"\nBy year:")
    for year, count in sorted(years.items()):
        print(f"  FY{year}: {count}")

    # Token stats
    tokens = [c["token_count"] for c in all_chunks]
    print(f"\nToken counts:")
    print(f"  Min: {min(tokens)}")
    print(f"  Max: {max(tokens)}")
    print(f"  Avg: {sum(tokens)/len(tokens):.0f}")

    # Sample chunks
    print(f"\n" + "="*60)
    print("SAMPLE CHUNKS")
    print("="*60)

    # Show one of each type
    for chunk_type in ["lead_sentence", "bullet_group", "paragraph"]:
        samples = [c for c in all_chunks if c["chunk_type"] == chunk_type][:1]
        for s in samples:
            print(f"\n[{s['chunk_id']}] {chunk_type} | {s['section'][:40]}...")
            print(f"  Tokens: {s['token_count']}")
            print(f"  Content: \"{s['content'][:150]}...\"")

    print(f"\nOutput saved to: {output_path}/")


if __name__ == "__main__":
    main()

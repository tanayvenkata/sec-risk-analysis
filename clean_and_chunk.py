#!/usr/bin/env python3
"""Clean and chunk Meta 10-K Risk Factors for RAG indexing."""

import json
import os
import re
from pathlib import Path

INPUT_DIR = "sec_corpus/META"
OUTPUT_DIR = "sec_corpus/META/processed"

# HTML entities to decode
HTML_ENTITIES = {
    "&#8226;": "•",
    "&#38;": "&",
    "&#8217;": "'",
    "&#8220;": '"',
    "&#8221;": '"',
    "&#8212;": "—",
    "&#160;": " ",
    "&amp;": "&",
    "&quot;": '"',
    "&apos;": "'",
    "&lt;": "<",
    "&gt;": ">",
}

# Section header patterns (these are the main risk categories)
SECTION_HEADERS = [
    "Risks Related to Our Product Offerings",
    "Risks Related to Our Business Operations and Financial Results",
    "Risks Related to Business Operations and Financial Results",
    "Risks Related to Government Regulation and Enforcement",
    "Risks Related to Data, Security, Platform Integrity, and Intellectual Property",
    "Risks Related to Data, Security, and Intellectual Property",
    "Risks Related to Ownership of Our Class A Common Stock",
    "Summary Risk Factors",
]


def clean_text(text: str) -> str:
    """Clean HTML entities and artifacts from extracted text."""
    # Decode HTML entities
    for entity, char in HTML_ENTITIES.items():
        text = text.replace(entity, char)

    # Remove "Table of Contents" artifacts
    text = re.sub(r"^\s*Table of Contents\s*$", "", text, flags=re.MULTILINE)

    # Normalize whitespace (collapse multiple newlines to max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def detect_section(line: str, current_section: str) -> str:
    """Detect if a line is a section header, return section name."""
    line_clean = line.strip()
    for header in SECTION_HEADERS:
        if line_clean == header:
            return header
    return current_section


def is_bullet_item(line: str) -> bool:
    """Check if line is a bullet point."""
    return line.strip().startswith("•")


def is_header_line(line: str) -> bool:
    """Check if line is a section header."""
    return line.strip() in SECTION_HEADERS or line.strip() == "Item 1A. Risk Factors"


def chunk_text(text: str, company: str, fiscal_year: int, filed_at: str) -> list[dict]:
    """
    Chunk cleaned text into structured pieces with metadata.

    Strategy:
    - Each non-empty line becomes a chunk
    - Bullet items are marked as such
    - Section headers are tracked but not chunked separately
    - Parent section is attached to each chunk
    """
    chunks = []
    current_section = "Introduction"
    chunk_index = 0

    lines = text.split("\n")

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check for section header
        new_section = detect_section(line, current_section)
        if new_section != current_section:
            current_section = new_section
            # Don't create a chunk for the header itself
            if is_header_line(line):
                continue

        # Skip the main title
        if line == "Item 1A. Risk Factors":
            continue

        # Skip standalone section headers
        if is_header_line(line):
            continue

        # Create chunk
        chunk = {
            "chunk_id": f"{company}_FY{fiscal_year}_{chunk_index:04d}",
            "company": company,
            "fiscal_year": fiscal_year,
            "filed_at": filed_at,
            "section": current_section,
            "content": line,
            "is_bullet": is_bullet_item(line),
            "word_count": len(line.split()),
            "char_count": len(line),
        }

        chunks.append(chunk)
        chunk_index += 1

    return chunks


def process_file(filepath: Path, metadata: dict) -> list[dict]:
    """Process a single risk factors file."""
    # Find metadata for this file
    file_meta = None
    for m in metadata["filings"]:
        if m["output_file"] == filepath.name:
            file_meta = m
            break

    if not file_meta:
        print(f"  Warning: No metadata found for {filepath.name}")
        return []

    # Read and clean
    raw_text = filepath.read_text(encoding="utf-8")
    clean = clean_text(raw_text)

    # Chunk
    chunks = chunk_text(
        clean,
        company=file_meta["ticker"],
        fiscal_year=file_meta["fiscal_year"],
        filed_at=file_meta["filed_at"],
    )

    return chunks


def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_file = input_path / "metadata.json"
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
    print(f"Bullet items: {sum(1 for c in all_chunks if c['is_bullet'])}")
    print(f"Paragraphs: {sum(1 for c in all_chunks if not c['is_bullet'])}")

    # Section breakdown
    sections = {}
    for c in all_chunks:
        sections[c["section"]] = sections.get(c["section"], 0) + 1

    print(f"\nChunks by section:")
    for section, count in sorted(sections.items(), key=lambda x: -x[1]):
        print(f"  {section}: {count}")

    # Year breakdown
    years = {}
    for c in all_chunks:
        years[c["fiscal_year"]] = years.get(c["fiscal_year"], 0) + 1

    print(f"\nChunks by year:")
    for year, count in sorted(years.items()):
        print(f"  FY{year}: {count}")

    print(f"\nOutput saved to: {output_path}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Clean 10-K text files (Risk Factors and MD&A) for multiple companies."""

import re
from pathlib import Path

BASE_DIR = "sec_corpus"

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


def clean_text(text: str) -> str:
    """Clean HTML entities and artifacts from extracted text."""
    # Decode HTML entities
    for entity, char in HTML_ENTITIES.items():
        text = text.replace(entity, char)

    # Remove "Table of Contents" artifacts (including "T able of Contents" with space)
    text = re.sub(r"^\s*T\s*able of Contents\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)

    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split("\n")]

    # Collapse blank lines between consecutive bullets so they group together
    # This ensures bullets aren't split into separate chunks
    result_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        result_lines.append(line)

        # If this is a bullet line, skip blank lines until next bullet or non-blank
        if line.startswith("•"):
            i += 1
            # Skip blank lines between bullets
            while i < len(lines) and lines[i] == "":
                # Peek ahead - if next non-blank is also a bullet, skip this blank
                next_non_blank = None
                for j in range(i, len(lines)):
                    if lines[j] != "":
                        next_non_blank = lines[j]
                        break
                if next_non_blank and next_non_blank.startswith("•"):
                    i += 1  # Skip the blank line
                else:
                    break  # Keep the blank line (end of bullet section)
        else:
            i += 1

    text = "\n".join(result_lines)

    # Normalize whitespace (collapse multiple newlines to max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def process_company(company_dir: Path):
    """Process all text files for a single company."""
    output_dir = company_dir / "cleaned"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all FY*.txt files (Risk Factors and MDA)
    txt_files = list(company_dir.glob("FY*.txt"))
    if not txt_files:
        print(f"  No text files found in {company_dir}")
        return

    for txt_file in sorted(txt_files):
        print(f"  Cleaning {txt_file.name}...")

        raw_text = txt_file.read_text(encoding="utf-8")
        cleaned = clean_text(raw_text)

        # Save cleaned version
        output_file = output_dir / txt_file.name
        output_file.write_text(cleaned, encoding="utf-8")

        # Stats
        raw_size = len(raw_text)
        clean_size = len(cleaned)
        print(f"    {raw_size:,} → {clean_size:,} chars ({100*clean_size/raw_size:.1f}%)")

    print(f"  Cleaned files saved to: {output_dir}/")


def main():
    base_path = Path(BASE_DIR)

    # Iterate over company directories
    for company_dir in sorted(base_path.iterdir()):
        # Skip non-directories and special directories
        if not company_dir.is_dir() or company_dir.name.startswith("_"):
            continue

        print(f"\nProcessing {company_dir.name}...")
        process_company(company_dir)


if __name__ == "__main__":
    main()

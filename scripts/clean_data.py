#!/usr/bin/env python3
"""Clean Meta 10-K Risk Factors text (no chunking yet)."""

import re
from pathlib import Path

INPUT_DIR = "sec_corpus/META"
OUTPUT_DIR = "sec_corpus/META/cleaned"

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


def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    for txt_file in sorted(input_path.glob("FY*.txt")):
        print(f"Cleaning {txt_file.name}...")

        raw_text = txt_file.read_text(encoding="utf-8")
        cleaned = clean_text(raw_text)

        # Save cleaned version
        output_file = output_path / txt_file.name
        output_file.write_text(cleaned, encoding="utf-8")

        # Stats
        raw_size = len(raw_text)
        clean_size = len(cleaned)
        print(f"  {raw_size:,} → {clean_size:,} chars ({100*clean_size/raw_size:.1f}%)")

    print(f"\nCleaned files saved to: {output_path}/")


if __name__ == "__main__":
    main()

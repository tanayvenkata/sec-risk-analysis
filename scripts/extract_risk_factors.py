#!/usr/bin/env python3
"""Extract Meta 10-K Risk Factors (Item 1A) for FY2020-2024 using sec-api."""

import json
import os
from sec_api import QueryApi, ExtractorApi

API_KEY = os.environ.get("SEC_API_KEY")
if not API_KEY:
    raise ValueError("SEC_API_KEY environment variable is required. Get one at https://sec-api.io")

# Initialize APIs
queryApi = QueryApi(api_key=API_KEY)
extractorApi = ExtractorApi(api_key=API_KEY)

# Output directory
OUTPUT_DIR = "sec_corpus/META"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Query for Meta's 10-K filings (both META and FB tickers)
print("Querying for Meta's 10-K filings...")
query = {
    "query": {
        "query_string": {
            "query": '(ticker:META OR ticker:FB) AND formType:"10-K"'
        }
    },
    "from": "0",
    "size": "10",
    "sort": [{"filedAt": {"order": "desc"}}]
}

response = queryApi.get_filings(query)
filings = response.get("filings", [])

print(f"\nFound {len(filings)} 10-K filings:\n")
for f in filings:
    print(f"  {f['filedAt'][:10]} | {f['ticker']} | {f['formType']} | {f['documentFormatFiles'][0]['documentUrl'][:80]}...")

# Filter to get one filing per fiscal year (2020-2024)
# 10-K filed in early year X is for fiscal year X-1
fiscal_year_filings = {}
for f in filings:
    filed_year = int(f['filedAt'][:4])
    fiscal_year = filed_year - 1  # 10-K filed in Feb 2024 is for FY2023

    if 2020 <= fiscal_year <= 2024:
        if fiscal_year not in fiscal_year_filings:
            fiscal_year_filings[fiscal_year] = f

print(f"\nFilings by fiscal year:")
for fy in sorted(fiscal_year_filings.keys()):
    f = fiscal_year_filings[fy]
    print(f"  FY{fy}: Filed {f['filedAt'][:10]}")

# Extract Risk Factors from each filing
metadata = {"filings": []}
preview_shown = False

print("\n" + "="*60)
print("Extracting Risk Factors (Item 1A) from each filing...")
print("="*60)

for fy in sorted(fiscal_year_filings.keys()):
    f = fiscal_year_filings[fy]
    filing_url = f['documentFormatFiles'][0]['documentUrl']

    print(f"\nExtracting FY{fy}...")

    # Extract Item 1A (Risk Factors) as text
    risk_factors = extractorApi.get_section(filing_url, "1A", "text")

    # Save to file
    filename = f"FY{fy}_risk_factors.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as out:
        out.write(risk_factors)

    file_size = os.path.getsize(filepath)
    print(f"  Saved: {filename} ({file_size:,} bytes)")

    # Add to metadata
    metadata["filings"].append({
        "fiscal_year": fy,
        "filed_at": f['filedAt'],
        "ticker": f['ticker'],
        "filing_url": filing_url,
        "output_file": filename,
        "file_size_bytes": file_size
    })

    # Show preview for first file
    if not preview_shown:
        print(f"\n--- Preview of FY{fy} Risk Factors (first 2000 chars) ---")
        print(risk_factors[:2000])
        print("...")
        print("--- End Preview ---")
        preview_shown = True

# Save metadata
metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
with open(metadata_path, "w", encoding="utf-8") as out:
    json.dump(metadata, out, indent=2)

print(f"\nMetadata saved to: {metadata_path}")

# Summary
print("\n" + "="*60)
print("CORPUS SUMMARY")
print("="*60)
total_size = sum(f["file_size_bytes"] for f in metadata["filings"])
print(f"Total files: {len(metadata['filings'])}")
print(f"Total corpus size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
print(f"\nFiles created in {OUTPUT_DIR}/:")
for f in metadata["filings"]:
    print(f"  {f['output_file']}: {f['file_size_bytes']:,} bytes")

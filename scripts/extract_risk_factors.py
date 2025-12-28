#!/usr/bin/env python3
"""Extract 10-K sections for multiple companies (Item 1A, Item 7) using sec-api."""

import json
import os
from dotenv import load_dotenv
from sec_api import QueryApi, ExtractorApi

load_dotenv()

API_KEY = os.environ.get("SEC_API_KEY")
if not API_KEY:
    raise ValueError("SEC_API_KEY environment variable is required. Get one at https://sec-api.io")

# Initialize APIs
queryApi = QueryApi(api_key=API_KEY)
extractorApi = ExtractorApi(api_key=API_KEY)

# Load config from config.json
CONFIG_FILE = "config.json"
with open(CONFIG_FILE) as f:
    config = json.load(f)

# Convert config format to internal format
TARGETS = []
for target in config.get("extraction_targets", []):
    TARGETS.append({
        "ticker": target["ticker"],
        "years": target["years"],
        "sections": list(target["sections"].keys())  # ["1A", "7"]
    })

# Section ID to name mapping (from config)
SECTION_NAMES = {}
for target in config.get("extraction_targets", []):
    for section_id, section_name in target["sections"].items():
        SECTION_NAMES[section_id] = section_name

# Output directory
BASE_OUTPUT_DIR = "sec_corpus"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

def process_target(target):
    ticker = target["ticker"]
    print(f"\nProcessing {ticker}...")
    
    # Create company dir
    company_dir = os.path.join(BASE_OUTPUT_DIR, ticker)
    os.makedirs(company_dir, exist_ok=True)
    
    # Query for 10-Ks
    query = {
        "query": {
            "query_string": {
                "query": f'ticker:{ticker} AND formType:"10-K"'
            }
        },
        "from": "0",
        "size": "20",
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    
    response = queryApi.get_filings(query)
    filings = response.get("filings", [])
    
    # Map filings to fiscal years
    fiscal_year_filings = {}
    for f in filings:
        filed_year = int(f['filedAt'][:4])
        # Heuristic: 10-K filed in early year X is for FY(X-1)
        # Apple fiscal year ends in Sept, so 10-K filed in Oct/Nov 2024 is for FY2024.
        # Meta fiscal year ends in Dec, filed in Feb 2025 is for FY2024.
        
        # Adjust logic based on ticker if needed, but standardizing on "filed_year - 1" 
        # is a safe approximation for "Fiscal Year" label for searching, even if technically off by a month.
        # For simplicity in this tool, we treat "Filing Year - 1" as the target Fiscal Year bucket.
        fiscal_year = filed_year - 1 
        
        if fiscal_year in target["years"]:
            if fiscal_year not in fiscal_year_filings:
                fiscal_year_filings[fiscal_year] = f
                
    # Extract sections
    metadata_file = os.path.join(company_dir, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file) as f:
            metadata = json.load(f)
    else:
        metadata = {"filings": []}
        
    for fy in sorted(fiscal_year_filings.keys()):
        f = fiscal_year_filings[fy]
        filing_url = f['documentFormatFiles'][0]['documentUrl']
        
        for section in target["sections"]:
            section_name = "Risk Factors" if section == "1A" else "MDA"
            filename = f"FY{fy}_{section_name.replace(' ', '_')}.txt"
            filepath = os.path.join(company_dir, filename)
            
            # Skip if already exists
            if os.path.exists(filepath):
                print(f"  Skipping FY{fy} {section_name} (already exists)")
                continue
                
            print(f"  Extracting FY{fy} {section_name}...")
            try:
                text = extractorApi.get_section(filing_url, section, "text")
                with open(filepath, "w", encoding="utf-8") as out:
                    out.write(text)
                    
                metadata["filings"].append({
                    "fiscal_year": fy,
                    "filed_at": f['filedAt'],
                    "ticker": ticker,
                    "filing_url": filing_url,
                    "section": section_name, # "Risk Factors" or "MDA"
                    "section_id": section,   # "1A" or "7"
                    "output_file": filename,
                    "file_size_bytes": os.path.getsize(filepath)
                })
            except Exception as e:
                print(f"    Error extraction {section}: {e}")

    # Save metdata
    with open(metadata_file, "w", encoding="utf-8") as out:
        json.dump(metadata, out, indent=2)

def main():
    for target in TARGETS:
        process_target(target)

if __name__ == "__main__":
    main()

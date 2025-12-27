# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SEC filing corpus builder for Meta (META/FB) 10-K Risk Factors sections. Uses sec-api.io to query and extract Item 1A from annual filings.

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key (get free key at https://sec-api.io)
export SEC_API_KEY=your_key_here
```

## Commands

```bash
# Run the extraction script
python extract_risk_factors.py
```

## Architecture

- `extract_risk_factors.py` - Main script that:
  1. Queries sec-api for Meta's 10-K filings (searches both META and FB tickers since ticker changed in 2022)
  2. Extracts Item 1A (Risk Factors) as plain text using ExtractorApi
  3. Saves each fiscal year to `sec_corpus/META/FY{year}_risk_factors.txt`
  4. Creates `metadata.json` with filing dates, URLs, and file sizes

- `sec_corpus/META/` - Output directory containing extracted Risk Factors text files

## SEC Filing Notes

- 10-K filings are filed in early year X for fiscal year X-1 (e.g., filed Feb 2024 = FY2023)
- Meta changed ticker from FB to META in June 2022

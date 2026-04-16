"""
SEC EDGAR filing downloader (robust version)

Fixes:
- Stable ticker → CIK mapping
- Correct SEC headers
- Safe filing URL resolution (index.html fallback)
- No EFTS search dependency (avoids 403 issues)
"""

import time
import json
import requests
from pathlib import Path
from typing import Optional, List, Dict

BASE_URL = "https://www.sec.gov"
RATE_LIMIT_SLEEP = 0.12  # SEC ~10 req/sec safe limit

HEADERS = {
    "User-Agent": "FinetuningProject (vamsiyms@gmail.com)",
    "Accept": "application/json",
}


# -----------------------------
# 1. TICKER -> CIK
# -----------------------------
def get_company_cik(ticker: str) -> Optional[str]:
    """
    Reliable SEC ticker → CIK mapping.
    """
    url = "https://www.sec.gov/files/company_tickers.json"

    resp = requests.get(url, headers=HEADERS, timeout=20)
    time.sleep(RATE_LIMIT_SLEEP)

    if resp.status_code != 200:
        print("SEC ERROR:", resp.status_code, resp.text[:200])
        return None

    data = resp.json()
    ticker = ticker.upper()

    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker:
            return str(entry["cik_str"]).zfill(10)

    return None


# -----------------------------
# 2. GET 10-K FILINGS
# -----------------------------
def get_10k_filings(cik: str, limit: int = 5) -> List[Dict]:
    """
    Fetch recent 10-K filings metadata.
    """
    # url = f"{BASE_URL}/submissions/CIK{cik}.json"
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    
    resp = requests.get(url, headers=HEADERS, timeout=20)
    time.sleep(RATE_LIMIT_SLEEP)

    resp.raise_for_status()
    data = resp.json()

    filings = data.get("filings", {}).get("recent", {})

    form_types = filings.get("form", [])
    acc_numbers = filings.get("accessionNumber", [])
    filing_dates = filings.get("filingDate", [])
    primary_docs = filings.get("primaryDocument", [])

    results = []

    for i, form in enumerate(form_types):
        if form == "10-K":
            results.append({
                "accession_number": acc_numbers[i],
                "filing_date": filing_dates[i],
                "primary_document": primary_docs[i],
                "cik": cik
            })

        if len(results) >= limit:
            break

    return results


# -----------------------------
# 3. DOWNLOAD FILING (FIXED)
# -----------------------------
def download_filing_text(
    cik: str,
    accession_number: str,
    primary_doc: str,
    save_dir: Path
) -> Path:
    """
    Downloads SEC filing safely.

    FIX:
    - avoids relying on primaryDocument path
    - falls back to index.html (always exists)
    """

    acc_clean = accession_number.replace("-", "")

    base_url = (
        f"{BASE_URL}/Archives/edgar/data/"
        f"{int(cik)}/{acc_clean}/"
    )

    # fallback chain (SEC is inconsistent)
    urls = [
        base_url + primary_doc,
        base_url + "index.html",
    ]

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{cik}_{acc_clean}.html"

    if save_path.exists():
        print(f"  Already downloaded: {save_path.name}")
        return save_path

    last_error = None

    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS, stream=True, timeout=30)
            time.sleep(RATE_LIMIT_SLEEP)

            if resp.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"  Saved: {save_path.name} ({url})")
                return save_path

            last_error = f"{resp.status_code} {url}"

        except Exception as e:
            last_error = str(e)

    raise Exception(f"Failed to download filing {accession_number}: {last_error}")


# -----------------------------
# 4. MAIN PIPELINE
# -----------------------------
def download_sector(
    tickers: List[str],
    save_dir: Path,
    filings_per_company: int = 3
):
    """
    Downloads 10-K filings for a list of tickers.
    """

    all_metadata = []

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")

        cik = get_company_cik(ticker)

        if not cik:
            print(f"  CIK not found for {ticker}, skipping")
            continue

        filings = get_10k_filings(cik, limit=filings_per_company)

        print(f"  Found {len(filings)} 10-K filings")

        for filing in filings:
            path = download_filing_text(
                cik=filing["cik"],
                accession_number=filing["accession_number"],
                primary_doc=filing["primary_document"],
                save_dir=save_dir
            )

            filing["local_path"] = str(path)
            filing["ticker"] = ticker

            all_metadata.append(filing)

    # Save dataset index
    meta_path = save_dir / "metadata.json"

    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\nDone. Saved {len(all_metadata)} filings → {meta_path}")

    return all_metadata


# -----------------------------
# 5. RUN
# -----------------------------
if __name__ == "__main__":

    TICKERS = [
        "JPM",
        "GS",
        "MS",
        "BAC",
        "WFC",
        "CB",
        "MET",
        "PRU",
        "BLK",
        "SCHW",
    ]

    download_sector(
        tickers=TICKERS,
        save_dir=Path("data/raw"),
        filings_per_company=3
    )
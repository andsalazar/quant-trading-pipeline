#!/usr/bin/env python3
"""
Macro Events Fetcher
====================

Purpose:
- Fetch economic calendar events from BLS.gov
- Add FOMC meeting dates from Federal Reserve calendar
- Update 00_06_macroevents_fomc.csv with new events

Data Sources:
- BLS.gov economic release schedule (web scraping)
- FOMC meeting dates (hardcoded from Federal Reserve calendar)

Called by: 00_run_daily_pipeline.py (STEP 6a)
Created: February 2026
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import os
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(BASE_PATH, "00_06_macroevents_fomc.csv")

# ============================================================================
# FOMC MEETING DATES (from Federal Reserve calendar)
# Update annually when the Fed publishes the next year's schedule
# ============================================================================

FOMC_DATES = {
    2024: [
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18"
    ],
    2025: [
        "2025-01-29", "2025-03-19", "2025-04-30", "2025-06-11",
        "2025-07-30", "2025-09-17", "2025-11-06", "2025-12-17"
    ],
    2026: [
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-11-05", "2026-12-16"
    ],
}

# ============================================================================
# BLS EVENT CATEGORIZATION
# ============================================================================

MACRO_KEYWORDS = {
    # Jobs - High Impact
    "Employment Situation": "Jobs",
    "Nonfarm Payrolls": "Jobs",
    "Unemployment Rate": "Jobs",
    "Job Openings": "Jobs",
    "JOLTS": "Jobs",
    "State Employment": "Jobs",
    "Metropolitan Area Employment": "Jobs",
    "Business Employment Dynamics": "Jobs",
    # Wages
    "Employment Cost Index": "Wages",
    "Usual Weekly Earnings": "Wages",
    "Average Hourly Earnings": "Wages",
    # Inflation - High Impact
    "Consumer Price Index": "Inflation",
    "CPI": "Inflation",
    "Producer Price Index": "Inflation",
    "PPI": "Inflation",
    "Real Earnings": "Inflation",
    "Import and Export Price Index": "Inflation",
    "Import Price": "Inflation",
    "Export Price": "Inflation",
    # Productivity
    "Productivity and Costs": "Productivity",
    "Productivity and Costs by Industry": "Productivity",
    "Labor Productivity": "Productivity",
    # Labor Market
    "College Enrollment": "Labor Market",
    "Employment Characteristics of Families": "Labor Market",
    "Labor Force Statistics": "Labor Market",
}


def get_impact_score(title, category):
    """Assign impact level based on release importance"""
    title_lower = title.lower()

    high_impact = [
        "employment situation", "nonfarm payroll", "unemployment rate",
        "consumer price index", "cpi", "producer price index", "ppi",
        "job openings", "jolts", "employment cost index"
    ]
    medium_impact = [
        "real earnings", "productivity", "import price", "export price",
        "usual weekly earnings"
    ]

    if category == "FOMC":
        return "High"
    elif any(kw in title_lower for kw in high_impact):
        return "High"
    elif any(kw in title_lower for kw in medium_impact):
        return "Medium"
    else:
        return "Low"


def get_fomc_records():
    """Generate FOMC meeting records from hardcoded calendar"""
    records = []
    for year, dates in FOMC_DATES.items():
        for date_str in dates:
            records.append({
                "date": pd.to_datetime(date_str),
                "release_title": "FOMC Meeting - Interest Rate Decision",
                "category": "FOMC",
                "impact": "High",
                "year": year
            })
    return records


def fetch_bls_events(start_date):
    """Fetch BLS economic calendar events from BLS.gov"""
    records = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    today = datetime.now().date()
    years_to_fetch = sorted(set([today.year, today.year + 1]))

    for year in years_to_fetch:
        url = f"https://www.bls.gov/schedule/{year}/home.htm"
        print(f"  Fetching BLS schedule for {year}...")

        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            year_count = 0
            tables = soup.find_all("table")
            for table in tables:
                rows = table.find_all("tr")
                for row in rows[1:]:  # Skip header
                    cols = row.find_all("td")
                    if len(cols) < 3:
                        continue

                    date_str = cols[0].text.strip()
                    title = cols[2].text.strip()

                    for keyword, category in MACRO_KEYWORDS.items():
                        if keyword.lower() in title.lower():
                            try:
                                release_date = datetime.strptime(date_str, "%A, %B %d, %Y").date()
                                if release_date >= start_date:
                                    impact = get_impact_score(title, category)
                                    records.append({
                                        "date": pd.to_datetime(release_date),
                                        "release_title": title,
                                        "category": category,
                                        "impact": impact,
                                        "year": year
                                    })
                                    year_count += 1
                                break
                            except ValueError:
                                continue

            print(f"  OK {year}: {year_count} new events")

        except Exception as e:
            print(f"  WARNING: Failed to fetch {year} data: {e}")
            continue

    print(f"  OK Fetched {len(records)} total BLS events")
    return records


def main():
    """Main execution"""
    print("=" * 80)
    print("MACRO EVENTS FETCHER - BLS Economic Calendar + FOMC")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {os.path.basename(OUTPUT_CSV)}")
    print("=" * 80)

    # Load existing data
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV, parse_dates=["date"])
        print(f"\n  Existing data: {len(existing_df):,} events")
        print(f"  Date range: {existing_df['date'].min().date()} to {existing_df['date'].max().date()}")
        start_date = (existing_df['date'].max() + timedelta(days=1)).date()
    else:
        existing_df = pd.DataFrame()
        start_date = datetime(2014, 1, 1).date()
        print("\n  No existing data - starting fresh")

    print(f"  Fetching events from: {start_date}\n")

    # Fetch new BLS events
    bls_records = fetch_bls_events(start_date)

    # Get FOMC dates (only new ones)
    fomc_records = [r for r in get_fomc_records() if r['date'].date() >= start_date]
    print(f"  OK {len(fomc_records)} new FOMC dates")

    # Combine
    all_new = bls_records + fomc_records

    if all_new:
        new_df = pd.DataFrame(all_new)

        if not existing_df.empty:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df

        # Deduplicate
        before = len(combined)
        combined['date'] = pd.to_datetime(combined['date'])
        combined = combined.drop_duplicates(subset=['date', 'release_title'], keep='first')
        combined = combined.sort_values('date').reset_index(drop=True)
        removed = before - len(combined)

        if removed > 0:
            print(f"  OK Removed {removed} duplicates")

        # Save
        combined.to_csv(OUTPUT_CSV, index=False)
        print(f"\n  OK Saved {len(combined):,} events to {os.path.basename(OUTPUT_CSV)}")
        print(f"  OK Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")

        # Category summary
        cats = combined['category'].value_counts()
        for cat, count in cats.items():
            print(f"     {cat}: {count:,}")
    else:
        print(f"\n  OK No new events found - data is up to date")

    print("\n" + "=" * 80)
    print("OK Macro events fetch complete!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Export a trimmed backtest.db for CI use.

The full local backtest.db is ~4.9 GB (too large for GitHub Releases).
This script extracts only the tables needed for CI scoring pipelines:
  - Fundamentals (income, balance, cash flow, metrics) — for Compass/Moonshot scores
  - Historical grades — for analyst accuracy analysis
  - Recent prices (last 400 days) — for volatility and SMA200
  - Analyst accuracy tables — for website export

Excludes: 22M+ older price records, backtest_daily_scores, backtest_signals.
Output is typically ~300-400 MB, well under GitHub's 2 GB release asset limit.

Usage:
    python scripts/export_ci_backtest.py                  # Export to backtest_ci.db
    python scripts/export_ci_backtest.py --upload         # Export and upload to release
"""

import sqlite3
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DB = PROJECT_ROOT / 'backtest.db'
OUTPUT_DB = PROJECT_ROOT / 'backtest_ci.db'

# Tables to copy in full
FULL_TABLES = [
    'historical_income_statements',
    'historical_balance_sheets',
    'historical_cash_flows',
    'historical_key_metrics',
    'historical_grades',
    'analyst_accuracy',
    'analyst_sector_accuracy',
    'analyst_estimates_snapshot',
    'price_target_summary',
    'collection_progress',
    'historical_sentiment',
    'dcf_valuations',
]

# Tables to copy with date filter (recent data only)
RECENT_TABLES = {
    'historical_prices': "date >= date('now', '-400 days')",
}

# Tables to create as empty schema (so CI doesn't crash on missing tables)
SCHEMA_ONLY_TABLES = [
    'backtest_daily_scores',
    'backtest_signals',
]


def export():
    if not SOURCE_DB.exists():
        print(f"ERROR: Source database not found: {SOURCE_DB}")
        sys.exit(1)

    # Remove old output
    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()

    src = sqlite3.connect(str(SOURCE_DB))
    dst = sqlite3.connect(str(OUTPUT_DB))

    # Get all table schemas from source
    tables = src.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_schemas = {name: sql for name, sql in tables}

    # Copy full tables
    for table in FULL_TABLES:
        if table not in table_schemas:
            print(f"  SKIP {table} (not in source)")
            continue
        dst.execute(table_schemas[table])
        rows = src.execute(f"SELECT * FROM {table}").fetchall()
        if rows:
            cols = len(rows[0])
            placeholders = ','.join(['?'] * cols)
            dst.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
        print(f"  {table}: {len(rows):,} rows")

    # Copy recent-only tables
    for table, where in RECENT_TABLES.items():
        if table not in table_schemas:
            print(f"  SKIP {table} (not in source)")
            continue
        dst.execute(table_schemas[table])
        rows = src.execute(f"SELECT * FROM {table} WHERE {where}").fetchall()
        if rows:
            cols = len(rows[0])
            placeholders = ','.join(['?'] * cols)
            dst.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
        print(f"  {table}: {len(rows):,} rows (filtered: {where})")

    # Create empty schema-only tables
    for table in SCHEMA_ONLY_TABLES:
        if table not in table_schemas:
            print(f"  SKIP {table} (not in source)")
            continue
        dst.execute(table_schemas[table])
        print(f"  {table}: schema only (0 rows)")

    # Copy indexes
    indexes = src.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
    ).fetchall()
    for (sql,) in indexes:
        try:
            dst.execute(sql)
        except sqlite3.OperationalError:
            pass  # Table might not exist in output

    dst.commit()

    # Vacuum to compact
    print("\nCompacting database...")
    dst.execute("VACUUM")
    dst.close()
    src.close()

    size_mb = OUTPUT_DB.stat().st_size / (1024 * 1024)
    print(f"\nExported: {OUTPUT_DB} ({size_mb:.1f} MB)")
    return size_mb


def upload():
    import shutil
    import tempfile

    # CI expects the asset named "backtest.db", so copy to a temp dir with that name
    with tempfile.TemporaryDirectory() as tmpdir:
        upload_path = Path(tmpdir) / 'backtest.db'
        shutil.copy2(str(OUTPUT_DB), str(upload_path))

        print(f"\nUploading to 'data' release as backtest.db ({upload_path.stat().st_size / (1024**2):.1f} MB)...")
        result = subprocess.run(
            ['gh', 'release', 'upload', 'data', str(upload_path), '--clobber'],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"ERROR: Upload failed: {result.stderr}")
            sys.exit(1)
    print("Upload complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export trimmed backtest.db for CI')
    parser.add_argument('--upload', action='store_true', help='Upload to GitHub release after export')
    args = parser.parse_args()

    print(f"Source: {SOURCE_DB} ({SOURCE_DB.stat().st_size / (1024**3):.1f} GB)")
    print(f"Output: {OUTPUT_DB}\n")

    size_mb = export()

    if size_mb > 1800:
        print(f"\nWARNING: Output is {size_mb:.0f} MB — approaching GitHub's 2 GB limit")

    if args.upload:
        upload()

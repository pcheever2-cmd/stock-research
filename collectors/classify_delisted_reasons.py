#!/usr/bin/env python3
"""
Classify delist reasons for the `delisted_companies` table via web research.

Three-stage pipeline (the middle stage is a Workflow of research agents):
  1. --prep    : write pending names to /tmp/delist_batches/batch_NNN.json (N names each),
                 print the batch count for the Workflow.
  2. (Workflow): one agent per batch reads its file, web-researches each name, and writes
                 /tmp/delist_results/result_NNN.json (same schema).
  3. --ingest  : load all result files, update delisted_companies (reason, detail, source,
                 confidence, classified_at). Reports coverage + reason distribution.

Reason set: acquired | went_private | merged | bankrupt | distress_delist | liquidated |
            not_operating | not_delisted | unknown
('not_delisted' = ticker rename / still trading — must NOT get a terminal delist return.)

Usage:
    python collectors/classify_delisted_reasons.py --prep [--batch-size 20]
    python collectors/classify_delisted_reasons.py --ingest
"""
import sys, json, argparse, sqlite3, glob, os
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BACKTEST_DB

BATCH_DIR = '/tmp/delist_batches'
RESULT_DIR = '/tmp/delist_results'
VALID_REASONS = {'acquired', 'went_private', 'merged', 'bankrupt', 'distress_delist',
                 'liquidated', 'not_operating', 'not_delisted', 'unknown'}


def prep(batch_size):
    conn = sqlite3.connect(str(BACKTEST_DB))
    rows = conn.execute(
        "SELECT symbol, company_name, delisted_date FROM delisted_companies "
        "WHERE reason IS NULL ORDER BY symbol").fetchall()
    conn.close()
    os.makedirs(BATCH_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    names = [{'symbol': s, 'name': n, 'delisted_date': d} for s, n, d in rows]
    nb = 0
    for i in range(0, len(names), batch_size):
        with open(f'{BATCH_DIR}/batch_{nb:04d}.json', 'w') as f:
            json.dump(names[i:i + batch_size], f)
        nb += 1
    print(f"Pending: {len(names)} names -> {nb} batch files of ~{batch_size} in {BATCH_DIR}")
    print(f"NUM_BATCHES={nb}")


def ingest():
    conn = sqlite3.connect(str(BACKTEST_DB))
    files = sorted(glob.glob(f'{RESULT_DIR}/result_*.json'))
    now = datetime.now(timezone.utc).isoformat()
    updated, bad, dist = 0, 0, {}
    for fp in files:
        try:
            recs = json.load(open(fp))
        except Exception:
            print(f"  ! unparseable: {fp}"); continue
        for r in recs:
            sym = r.get('symbol'); reason = (r.get('reason') or '').strip()
            if not sym or reason not in VALID_REASONS:
                bad += 1; continue
            conn.execute(
                "UPDATE delisted_companies SET reason=?, reason_detail=?, reason_source=?, "
                "reason_confidence=?, classified_at=? WHERE symbol=?",
                (reason, (r.get('detail') or '')[:120], r.get('source'),
                 r.get('confidence'), now, sym))
            updated += conn.total_changes and 1 or 0
            dist[reason] = dist.get(reason, 0) + 1
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM delisted_companies").fetchone()[0]
    classified = conn.execute("SELECT COUNT(*) FROM delisted_companies WHERE reason IS NOT NULL").fetchone()[0]
    conn.close()
    print(f"Ingested {len(files)} result files | bad/skipped recs: {bad}")
    print(f"Classified {classified}/{total} ({total-classified} pending)")
    print("Reason distribution:", dict(sorted(dist.items(), key=lambda x: -x[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', action='store_true')
    ap.add_argument('--ingest', action='store_true')
    ap.add_argument('--batch-size', type=int, default=20)
    args = ap.parse_args()
    if args.prep:
        prep(args.batch_size)
    elif args.ingest:
        ingest()
    else:
        ap.error("pass --prep or --ingest")


if __name__ == '__main__':
    main()

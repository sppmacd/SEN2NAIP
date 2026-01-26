#!/usr/bin/env python3
import csv
import subprocess
import sys
from pathlib import Path

INFILE = Path('experiments-t.csv')
BACKUP = INFILE.with_suffix('.csv.bak')

if not INFILE.exists():
    print(f"Input file {INFILE} not found", file=sys.stderr)
    sys.exit(2)

with INFILE.open('r', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

missing = []
for r in rows:
    rev = (r.get('rev') or '').strip()
    if not rev:
        r['_ts'] = 0
        continue
    try:
        out = subprocess.check_output(['git', 'show', '-s', '--format=%ct', rev], stderr=subprocess.DEVNULL)
        ts = int(out.decode().strip())
        r['_ts'] = ts
    except Exception:
        r['_ts'] = 0
        missing.append(rev)

# sort newest first
rows.sort(key=lambda x: x.get('_ts', 0), reverse=True)

# backup
INFILE.rename(BACKUP)

with INFILE.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        out = {k: r.get(k, '') for k in fieldnames}
        writer.writerow(out)

print(f"Wrote sorted {INFILE} (backup at {BACKUP}).")
if missing:
    uniq = sorted(set(missing))
    print("Revisions not found in git:")
    for u in uniq:
        print(f" - {u}")

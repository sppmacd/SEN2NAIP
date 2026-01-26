#!/usr/bin/env python3
"""
Generate a compact experiments table from experiments.csv.

Selects these columns (if present):
  - rev
  - dvclive/train/metrics.json:step   -> shown as `step`
  - test_ssim
  - test_psnr
  - train_time
  - train.mode
  - train.batch_size
  - train.gradient_accumulation_steps
  - train.optimizer
  - train.loss
  - train.max_epochs

Adds `description` by preferring `Experiment` column or reading git commit title for `rev`.

Writes CSV to stdout or to `--out` file.
"""

import argparse
import csv
import subprocess
import sys
import math
from datetime import datetime
from pathlib import Path


WANTED = [
    "rev",
    "dvclive/train/metrics.json:step",
    "test_ssim",
    "test_psnr",
    "train_time",
    "train.mode",
    "train.batch_size",
    "train.gradient_accumulation_steps",
    "train.optimizer",
    "train.loss",
    "train.max_epochs",
]

# human-friendly header replacements (key = cleaned header name)
HEADER_REPLACEMENTS = {
    "train_time": "Train time (min)",
}


def _format_out_header(name: str) -> str:
    # apply replacement and wrap in asterisks
    rep = HEADER_REPLACEMENTS.get(name, name)
    return rep


def find_column(header_names, wanted_key):
    # Exact match first
    if wanted_key in header_names:
        return wanted_key
    # For json-like keys, match by suffix after ':'
    if ":" in wanted_key:
        suffix = wanted_key.split(":", 1)[-1]
        for h in header_names:
            if h.endswith(":" + suffix) or h == suffix:
                return h
    # fallback: if simple name (no path) try direct match
    for h in header_names:
        if h == wanted_key:
            return h
    return None


def git_commit_title(rev, repo_dir: Path):
    if not rev:
        return ""
    try:
        out = subprocess.run(
            ["git", "show", "-s", "--format=%s", rev],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def clean_header(h: str) -> str:
    # Remove json file names / path prefixes; keep last token after ':' if present
    if ":" in h:
        return h.split(":", 1)[-1]
    # strip leading 'train.' prefix if present
    if h.startswith("train."):
        return h.split(".", 1)[1]
    return h


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in", dest="infile", default="experiments.csv", help="Input CSV path"
    )
    p.add_argument(
        "--out", dest="outfile", default=None, help="Output CSV path (default stdout)"
    )
    p.add_argument(
        "--repo", dest="repo", default=".", help="Path to git repo (for commit titles)"
    )
    args = p.parse_args(argv)

    infile = Path(args.infile)
    if not infile.exists():
        print(f"Input file not found: {infile}", file=sys.stderr)
        return 2

    with infile.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # Resolve the columns we will output (map wanted key -> actual column name)
        resolved = []
        for w in WANTED:
            col = find_column(headers, w)
            resolved.append((w, col))

        # Build output header names (cleaned) and insert description after rev
        out_headers = []
        # we will combine batch_size and gradient_accumulation_steps into single "batch_size"
        for w, col in resolved:
            if w == "train.batch_size":
                base = "batch_size"
            elif w == "train.gradient_accumulation_steps":
                # skip, merged into previous column
                continue
            else:
                base = clean_header(w) if col is None else clean_header(col)

            out_headers.append(_format_out_header(base))
            # insert description immediately after rev
            if w == "rev":
                out_headers.append(_format_out_header("description"))

        # Prepare writer
        if args.outfile:
            out_f = open(args.outfile, "w", newline="", encoding="utf-8")
        else:
            out_f = sys.stdout

        writer = csv.writer(out_f)
        writer.writerow(out_headers)

        repo_dir = Path(args.repo)
        # prepare quick lookup for resolved columns
        resolved_map = {w: col for w, col in resolved}

        # resolve experiment column (if present) to prefer as description
        experiment_col = find_column(headers, "Experiment")

        # collect rows then sort by mode
        rows = []
        mode_col = resolved_map.get("train.mode")

        # locate train_time and Created columns for numeric sorting
        train_time_col = resolved_map.get("train_time")
        created_col = find_column(headers, "Created")

        for row in reader:
            out_row = []
            # precompute rev/description so we can insert description after rev
            rev_col = resolved_map.get("rev") or find_column(headers, "rev")
            rev_val = row.get(rev_col, "") if rev_col else ""
            commit_title = git_commit_title(rev_val, repo_dir)
            exp_val = row.get(experiment_col, "") if experiment_col else ""
            description = exp_val if exp_val else commit_title
            for w, col in resolved:
                if w == "train.gradient_accumulation_steps":
                    # handled together with batch_size; skip
                    continue

                if w == "train.batch_size":
                    bs_col = col
                    ga_col = resolved_map.get("train.gradient_accumulation_steps")
                    bs_val = row.get(bs_col, "") if bs_col else ""
                    ga_val = row.get(ga_col, "") if ga_col else ""
                    if bs_val and ga_val:
                        val = f"{ga_val} x {bs_val}"
                    else:
                        # fallback to whichever present
                        val = bs_val or ga_val or ""
                    out_row.append(val)
                    continue

                if w == "rev":
                    # insert rev value then description as second column
                    val = rev_val
                    out_row.append(val)
                    out_row.append(description)
                    continue

                if col is None:
                    val = ""
                else:
                    val = row.get(col, "")
                    # round selected numeric metrics to 2 decimals
                    if w in ("test_ssim", "test_psnr") and val:
                        try:
                            f = float(val)
                            if w == "test_psnr":
                                # PSNR was computed with MAX=255; convert to MAX=1
                                f = f - 20 * math.log10(255)
                            elif w == "test_ssim":
                                # display as -log(1-ssim)
                                s = 1.0 - f
                                if s <= 0.0:
                                    f = float("inf")
                                else:
                                    f = -math.log(s)
                            val = f"{f:.2f}"
                        except Exception:
                            pass
                    elif w in ("train_time",) and val:
                        try:
                            f = float(val)
                            val = f"{f / 60:.2f}"
                        except Exception:
                            pass
                    # increment step by 1 if this is a step column
                    elif w.endswith(":step") and val:
                        try:
                            n = int(float(val))
                            val = str(n + 1)
                        except Exception:
                            pass
                out_row.append(val)

            mode_val = row.get(mode_col, "") if mode_col else ""
            # compute created timestamp for sorting (newest first)
            created_ts = 0.0
            if created_col:
                try:
                    raw_created = row.get(created_col, "")
                    if raw_created:
                        dt = datetime.fromisoformat(raw_created)
                        created_ts = dt.timestamp()
                except Exception:
                    created_ts = 0.0

            # compute numeric train_time in minutes for sorting
            tt_num = 0.0
            if train_time_col:
                try:
                    raw_tt = row.get(train_time_col, "")
                    if raw_tt:
                        tt_num = float(raw_tt) / 60.0
                except Exception:
                    tt_num = 0.0

            rows.append((mode_val, created_ts, tt_num, out_row))

        # sort rows: overfitting first, then empty mode, then others
        def mode_key(m):
            if m == "overfitting":
                return 0
            if m == "":
                return 1
            return 2

        # sort rows: mode priority, then Created (newest first), then train_time descending
        rows.sort(key=lambda x: (mode_key(x[0]), -x[1], -x[2], x[0]))

        for _, _, _, out_row in rows:
            writer.writerow(out_row)

        if args.outfile:
            out_f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

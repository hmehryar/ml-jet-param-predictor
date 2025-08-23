#!/usr/bin/env python3
"""
Mirror-copy a subset of files (e.g., ds1008) into a new directory,
preserving the original directory structure.

Example:
    python mirror_subset_copy.py \
        --original-root /path/to/ML-JET-7.2M \
        --csv file_labels_aggregated_ds1008_g500.csv \
        --dest-root /path/to/ML-JET-ds1008-mirror \
        --copy-mode copy --workers 16

Supported copy modes:
  - copy      : shutil.copy2 (default; preserves metadata)
  - link      : create hard links (fast, same filesystem)
  - symlink   : create symbolic links
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
from typing import Optional


# Common candidate column names for file paths inside the CSV
CANDIDATE_PATH_COLS = [
    "path", "filepath", "file_path", "image_path", "img_path",
    "relative_path", "relpath", "rel_path", "filename"
]

def choose_path_column(csv_path: Path, user_col: Optional[str]) -> str:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in reader.fieldnames or []]
    if user_col:
        if user_col not in header:
            raise ValueError(f"--path-column '{user_col}' not found in CSV. Columns: {header}")
        return user_col
    for cand in CANDIDATE_PATH_COLS:
        if cand in header:
            return cand
    raise ValueError(
        f"Could not infer a path column. Please pass --path-column. "
        f"Available columns: {header}"
    )
import json, re

MULTI_SPLIT_REGEX = re.compile(r"[|;,]\s*")  # handles | ; ,

def expand_paths(cell: str) -> list[str]:
    cell = (cell or "").strip()
    if not cell:
        return []
    # Try JSON first: ["p1","p2", ...]
    if cell.startswith("[") and cell.endswith("]"):
        try:
            arr = json.loads(cell)
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    # Fallback: split on | ; , (common in aggregated CSVs)
    if MULTI_SPLIT_REGEX.search(cell):
        return [p for p in MULTI_SPLIT_REGEX.split(cell) if p]
    # Single path
    return [cell]

def iter_csv_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def resolve_relative_path(original_root: Path, p: str) -> Path:
    p = p.strip()
    p_path = Path(p)
    # If absolute and under original_root, make it relative to original_root.
    try:
        if p_path.is_absolute():
            rel = p_path.relative_to(original_root)
        else:
            rel = p_path
    except ValueError:
        # Absolute path, but not under original_root — fall back to relpath heuristics
        rel = Path(os.path.relpath(p_path, start=original_root))
    return rel

def ensure_parent_dir(dest_file: Path):
    dest_file.parent.mkdir(parents=True, exist_ok=True)

def do_copy(src: Path, dst: Path, mode: str):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "link":
        # Hard link (fails across filesystems)
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")

def main():
    ap = argparse.ArgumentParser(description="Mirror-copy a subset of dataset files into a new root.")
    ap.add_argument("--original-root", required=True, type=Path,
                    help="Root directory of the full 7.2M dataset.")
    ap.add_argument("--csv", required=True, type=Path,
                    help="CSV file listing the 1008 aggregated sample file paths.")
    ap.add_argument("--dest-root", required=True, type=Path,
                    help="Destination root where the mirrored subset will be created.")
    ap.add_argument("--path-column", default=None, help="Column name in CSV that contains file paths.")
    ap.add_argument("--copy-mode", default="copy", choices=["copy", "link", "symlink"],
                    help="Copy strategy: 'copy' (default), 'link' (hard link), or 'symlink'.")
    ap.add_argument("--workers", type=int, default=8, help="Thread workers for parallel copy.")
    ap.add_argument("--dry-run", action="store_true", help="Print planned actions without writing.")
    args = ap.parse_args()

    original_root = args.original_root.resolve()
    dest_root = args.dest_root.resolve()
    csv_path = args.csv.resolve()

    if not original_root.exists():
        print(f"[ERROR] original-root not found: {original_root}", file=sys.stderr)
        sys.exit(1)
    if not csv_path.exists():
        print(f"[ERROR] csv not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if args.copy_mode in ("link", "symlink") and args.dry_run:
        print("[NOTE] dry-run with link/symlink will only print actions.")

    path_col = choose_path_column(csv_path, args.path_column)

    # Prepare destination root
    if not args.dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)

    planned = []
    missing = []
    duplicates = 0
    seen = set()

    # Gather all tasks
    for row in iter_csv_rows(csv_path):
        
        for raw_path in expand_paths(row[path_col]):
            if not raw_path:
                continue
            if raw_path in seen:
                duplicates += 1
                continue
            seen.add(raw_path)

            rel = resolve_relative_path(original_root, raw_path)
            src_file = (original_root / rel).resolve()

            if not src_file.exists():
                # Sometimes CSVs store paths relative to a subdir;
                # you can customize here if needed.
                missing.append(str(src_file))
                continue

            dest_file = dest_root / rel
            planned.append((src_file, dest_file))

    total = len(planned)
    print(f"[INFO] Files planned: {total} | Missing in source: {len(missing)} | Duplicates skipped: {duplicates}")
    if missing:
        # Print first few missing for debugging
        preview = "\n  ".join(missing[:10])
        print(f"[WARN] Missing examples (first 10):\n  {preview}")

    if args.dry_run:
        # Show a small sample of what would be done
        for i, (s, d) in enumerate(planned[:20], 1):
            print(f"[DRY-RUN] {i:04d}: {s}  ->  {d}")
        print("[DRY-RUN] Done.")
        return

    # Parallel copy
    copied = 0
    skipped = 0
    errors = 0

    def worker(task):
        src, dst = task
        try:
            if dst.exists():
                # already there — keep idempotent
                return ("skipped", src, dst, None)
            ensure_parent_dir(dst)
            do_copy(src, dst, args.copy_mode)
            return ("copied", src, dst, None)
        except Exception as e:
            return ("error", src, dst, e)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, t) for t in planned]
        for fut in as_completed(futures):
            status, src, dst, err = fut.result()
            if status == "copied":
                copied += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"[ERROR] {src} -> {dst}: {err}", file=sys.stderr)

    print(f"[DONE] copied={copied}, skipped={skipped}, missing_in_source={len(missing)}, errors={errors}")
    print(f"[OUT] Destination root: {dest_root}")

if __name__ == "__main__":
    main()

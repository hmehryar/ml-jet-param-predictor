#!/usr/bin/env python3
"""
make_folds.py

Generate 5 stratified folds (train/val CSVs) from the old validation set.
Stratification is done on the joint label (energy, alpha, q0) to preserve
the physics combinations.

Usage:
    python make_folds.py \
      --val_csv old_val.csv \
      --out_dir folds_out/ \
      --n_splits 5 \
      --seed 137
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv", required=True, help="Path to old validation CSV")
    parser.add_argument("--out_dir", required=True, help="Output directory for folds")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=137, help="Random seed for reproducibility")
    parser.add_argument("--col_energy", default="energy_loss")
    parser.add_argument("--col_alpha", default="alpha")
    parser.add_argument("--col_q0", default="q0")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.val_csv)

    # Build joint label for stratification
    df["joint_label"] = (
        df[args.col_energy].astype(str) + "_" +
        df[args.col_alpha].astype(str) + "_" +
        df[args.col_q0].astype(str)
    )

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["joint_label"])):
        train_csv = os.path.join(args.out_dir, f"fold{fold}_train.csv")
        val_csv   = os.path.join(args.out_dir, f"fold{fold}_val.csv")

        df.iloc[train_idx].drop(columns=["joint_label"]).to_csv(train_csv, index=False)
        df.iloc[val_idx].drop(columns=["joint_label"]).to_csv(val_csv, index=False)

        print(f"Fold {fold}: train={len(train_idx)} val={len(val_idx)}")
        print(f"  saved -> {train_csv}")
        print(f"  saved -> {val_csv}")

if __name__ == "__main__":
    main()

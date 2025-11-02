from __future__ import annotations

import argparse
from pathlib import Path

from autism_pipeline import generate_graphs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run autism EDA graphs and save interactive HTML files.")
    p.add_argument("--csv", dest="csv_path", type=str, default="autism_screening.csv",
                   help="Path to CSV dataset")
    p.add_argument("--out", dest="outdir", type=str, default="eda_outputs",
                   help="Output directory for HTML figures")
    p.add_argument("--graph", dest="graph_key", type=str, default="all",
                   help="Which graph to run: all | missing_counts | missing_heatmap | numeric_hists | numeric_boxes | corr_pearson | corr_spearman | cat_bars | stacked_bars | scatter | violin | strip")
    p.add_argument("--target", dest="target", type=str, default=None,
                   help="Target column name (optional, needed for some graphs)")
    p.add_argument("--seed", dest="seed", type=int, default=42)
    p.add_argument("--max-cat", dest="max_cat", type=int, default=50,
                   help="Max cardinality to treat as categorical")
    p.add_argument("--bins", dest="n_bins", type=int, default=30)
    p.add_argument("--max-cols", dest="max_cols", type=int, default=20)
    p.add_argument("--max-pairs", dest="max_pairs", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_graphs(
        csv_path=Path(args.csv_path),
        outdir=Path(args.outdir),
        graph_key=args.graph_key,
        target=args.target,
        seed=args.seed,
        max_cat_cardinality=args.max_cat,
        n_bins=args.n_bins,
        max_cols=args.max_cols,
        max_pairs=args.max_pairs,
    )
    print("Saved:")
    for p in paths:
        print("-", p)


if __name__ == "__main__":
    main()



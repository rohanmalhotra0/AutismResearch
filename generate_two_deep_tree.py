import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.patches as mpatches


def normalize_column_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def find_column(candidates, columns_norm_map):
    for cand in candidates:
        key = normalize_column_name(cand)
        if key in columns_norm_map:
            return columns_norm_map[key]
    return None


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Build a normalization map from normalized -> original
    columns_norm_map = {normalize_column_name(c): c for c in df.columns}

    # Identify label column
    label_col = find_column(
        ["Class/ASD", "class_asd", "classasd", "class", "target", "label"],
        columns_norm_map,
    )
    if label_col is None:
        raise RuntimeError("Could not locate the label column (e.g. 'Class/ASD').")

    # Identify A1..A10 columns
    a_cols = []
    for i in range(1, 11):
        c = find_column([f"A{i}_Score", f"a{i}_score", f"a{i}"], columns_norm_map)
        if c is None:
            raise RuntimeError(f"Missing required score column A{i}_Score.")
        a_cols.append(c)

    # Optional result column; if missing, compute as sum of A1..A10
    result_col = find_column(["Result", "result", "total", "sum"], columns_norm_map)
    if result_col is None:
        df["__result_sum__"] = df[a_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        result_col = "__result_sum__"

    # Build features: A1..A10 + result
    feat_cols = a_cols + [result_col]
    a_only_cols = a_cols[:]  # A1..A10 without 'result' for alternative tree
    df_feat = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df_feat_a_only = df[a_only_cols].apply(pd.to_numeric, errors="coerce")
    df_feat_a_only = df_feat_a_only.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Build label y as 0/1
    y_raw = df[label_col]
    def to_binary(v):
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "asd"}:
            return 1
        if s in {"0", "false", "no", "not asd", "non-asd", "none"}:
            return 0
        # fallback: try numeric
        try:
            f = float(s)
            return 1 if f >= 0.5 else 0
        except Exception:
            return 0
    y = y_raw.map(to_binary).astype(int).values

    return df_feat, df_feat_a_only, y, feat_cols, a_only_cols


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(root, "autism_screening.csv")
    if not os.path.isfile(csv_path):
        print(f"Dataset not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    X_df, X_df_a_only, y, feat_cols, a_only_cols = load_dataset(csv_path)
    X = X_df.values
    X_a = X_df_a_only.values

    clf = DecisionTreeClassifier(
        max_depth=2,
        criterion="entropy",  # ensures entropy impurity is computed/shown
        class_weight=None,    # use integer sample counts; no reweighting
        min_samples_leaf=10,  # avoid trivial/pure leaves; reflect real impurity
        random_state=42,
    )
    clf.fit(X, y)

    figures_dir = os.path.join(root, "figures")
    ensure_dir(figures_dir)
    out_path = os.path.join(figures_dir, "tree_two_deep.png")

    fig = plt.figure(figsize=(11, 6.5), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")
    artists = plot_tree(
        clf,
        feature_names=feat_cols,
        class_names=["No ASD", "ASD"],
        impurity=True,        # show entropy values
        proportion=False,     # show integer sample counts, not proportions
        filled=True,          # color-filled nodes
        rounded=True,         # rounded boxes for readability
        fontsize=9,
    )
    plt.title("Two Deep Decision Tree ", pad=8)

    # Add a simple legend clarifying class color mapping for medical context
    # Note: plot_tree uses a palette; we provide a consistent legend independent of node shades.
    # Blue = No ASD, Orange = ASD to match the rest of the report.
    legend_handles = [
        mpatches.Patch(color="#4C78A8", label="No ASD (majority color)"),
        mpatches.Patch(color="#F58518", label="ASD (majority color)"),
    ]
    leg = plt.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=8)
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")

    # Also generate a deeper tree (max_depth=10)
    clf10 = DecisionTreeClassifier(
        max_depth=10,
        criterion="entropy",
        class_weight=None,
        min_samples_leaf=10,
        random_state=42,
    )
    clf10.fit(X, y)
    out_path10 = os.path.join(figures_dir, "tree_depth10.png")
    fig = plt.figure(figsize=(14, 8), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")
    plot_tree(
        clf10,
        feature_names=feat_cols,
        class_names=["No ASD", "ASD"],
        impurity=True,
        proportion=False,
        filled=True,
        rounded=True,
        fontsize=7,
        max_depth=10,
    )
    plt.title("Decision Tree Depth=10", pad=8)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path10, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path10}")

    # Generate a deeper tree (depth=10) EXCLUDING 'result' to avoid a trivial single split
    clf10_nr = DecisionTreeClassifier(
        max_depth=10,
        criterion="entropy",
        class_weight=None,
        min_samples_leaf=5,
        random_state=42,
    )
    clf10_nr.fit(X_a, y)
    out_path10_nr = os.path.join(figures_dir, "tree_depth10_noresult.png")
    fig = plt.figure(figsize=(14, 9), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")
    plot_tree(
        clf10_nr,
        feature_names=a_only_cols,
        class_names=["No ASD", "ASD"],
        impurity=True,
        proportion=False,
        filled=True,
        rounded=True,
        fontsize=7,
        max_depth=10,
    )
    plt.title("Decision Tre Depth=10", pad=8)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path10_nr, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path10_nr}")


if __name__ == "__main__":
    main()


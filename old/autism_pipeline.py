from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import warnings

import plotly.express as px
import plotly.graph_objects as go


def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)


def infer_types(df: pd.DataFrame, max_cat_cardinality: int = 50) -> Tuple[List[str], List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*infer_datetime_format.*")
                warnings.filterwarnings("ignore", message="Could not infer format.*")
                parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().mean() > 0.9:
                datetime_cols.append(col)
            else:
                if s.nunique(dropna=True) <= max_cat_cardinality:
                    categorical_cols.append(col)
                else:
                    categorical_cols.append(col)
    return numeric_cols, categorical_cols, datetime_cols


def normalize_categorical_series(s: pd.Series) -> pd.Series:
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.astype(str).str.strip().str.lower().replace({"nan": np.nan})
    return s


def clean_dataframe(df: pd.DataFrame,
                    numeric_cols: List[str],
                    categorical_cols: List[str],
                    datetime_cols: List[str]) -> pd.DataFrame:
    df_clean = df.copy()
    for col in df_clean.columns.difference(numeric_cols):
        df_clean[col] = normalize_categorical_series(df_clean[col])
    for col in datetime_cols:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*infer_datetime_format.*")
            warnings.filterwarnings("ignore", message="Could not infer format.*")
            df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
    return df_clean


def build_data_dictionary(df: pd.DataFrame,
                          numeric_cols: List[str],
                          categorical_cols: List[str],
                          datetime_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        missing_pct = float(s.isna().mean() * 100)
        entry: Dict[str, object] = {"column": col, "dtype": dtype, "%missing": round(missing_pct, 2)}
        if col in numeric_cols:
            desc = s.describe(percentiles=[])
            entry.update({
                "min": float(desc.get("min", np.nan)),
                "mean": float(desc.get("mean", np.nan)),
                "std": float(desc.get("std", np.nan)),
                "max": float(desc.get("max", np.nan)),
            })
        elif col in categorical_cols:
            nunique = int(s.nunique(dropna=True))
            examples = s.dropna().astype(str).unique()[:5]
            entry.update({"#unique": nunique, "examples": ", ".join(map(str, examples))})
        elif col in datetime_cols:
            parsed = pd.to_datetime(s, errors="coerce")
            entry.update({
                "min": str(parsed.min()) if parsed.notna().any() else None,
                "max": str(parsed.max()) if parsed.notna().any() else None,
            })
        rows.append(entry)
    return pd.DataFrame(rows)


# ---------- Plot functions return (fig, suggested_filename) or list thereof ----------

def plot_missing_counts(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    missing_counts = df.isna().sum().sort_values(ascending=False)
    missing_df = missing_counts.reset_index()
    missing_df.columns = ["column", "missing_count"]
    fig = px.bar(missing_df, x="column", y="missing_count",
                 title="Missing Values per Column",
                 labels={"missing_count": "Missing Count", "column": "Column"})
    fig.update_layout(xaxis_tickangle=-45, height=450)
    return fig, "missing_counts.html"


def plot_missing_heatmap(df: pd.DataFrame, seed: int = 42) -> Tuple[go.Figure, str]:
    sample_n = min(500, len(df))
    ms_sample = df.sample(sample_n, random_state=seed)
    miss_bool = ms_sample.isna().astype(int)
    fig = px.imshow(miss_bool.T, aspect="auto",
                    color_continuous_scale=[(0.0, "#1a1a1a"), (1.0, "#e74c3c")],
                    title="Missingness Heatmap (1=Missing) - Transposed")
    fig.update_yaxes(title="Columns")
    fig.update_xaxes(title="Sample Rows")
    fig.update_coloraxes(colorbar_title_text="Missing")
    return fig, "missing_heatmap.html"


def plot_numeric_histograms(clean_df: pd.DataFrame, num_cols: List[str], n_bins: int = 30, max_cols: int = 20) -> List[Tuple[go.Figure, str]]:
    figs: List[Tuple[go.Figure, str]] = []
    for col in num_cols[:max_cols]:
        fig = px.histogram(clean_df, x=col, nbins=n_bins, marginal="violin",
                           title=f"Histogram + Violin for {col}", opacity=0.85)
        fig.update_traces(marker_color="#2ecc71")
        fig.update_layout(bargap=0.05)
        figs.append((fig, f"hist_violin__{col}.html"))
    return figs


def plot_numeric_boxplots(clean_df: pd.DataFrame, num_cols: List[str], max_cols: int = 20) -> List[Tuple[go.Figure, str]]:
    figs: List[Tuple[go.Figure, str]] = []
    for col in num_cols[:max_cols]:
        fig = px.box(clean_df, y=col, points="outliers", title=f"Boxplot (Outliers) for {col}")
        fig.update_traces(marker_color="#e67e22")
        figs.append((fig, f"boxplot__{col}.html"))
    return figs


def plot_correlations(clean_df: pd.DataFrame, num_cols: List[str]) -> List[Tuple[go.Figure, str]]:
    figs: List[Tuple[go.Figure, str]] = []
    if len(num_cols) >= 2:
        pearson_corr = clean_df[num_cols].corr(method="pearson")
        spearman_corr = clean_df[num_cols].corr(method="spearman")
        fig_cp = px.imshow(pearson_corr, color_continuous_scale="RdBu_r",
                           title="Pearson Correlation (Numeric Features)")
        fig_cp.update_xaxes(side="bottom")
        figs.append((fig_cp, "corr_pearson.html"))

        fig_cs = px.imshow(spearman_corr, color_continuous_scale="RdBu_r",
                           title="Spearman Correlation (Numeric Features)")
        fig_cs.update_xaxes(side="bottom")
        figs.append((fig_cs, "corr_spearman.html"))
    return figs


def plot_categorical_bars(clean_df: pd.DataFrame, cat_cols: List[str], max_cols: int = 20) -> List[Tuple[go.Figure, str]]:
    figs: List[Tuple[go.Figure, str]] = []
    for col in cat_cols[:max_cols]:
        vc = clean_df[col].value_counts(dropna=False).reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(vc.head(20), x=col, y="count", title=f"Top Categories for {col}")
        fig.update_layout(xaxis_tickangle=-45)
        figs.append((fig, f"cat_bars__{col}.html"))
    return figs


def plot_stacked_vs_target(clean_df: pd.DataFrame, cat_cols: List[str], target: Optional[str]) -> List[Tuple[go.Figure, str]]:
    figs: List[Tuple[go.Figure, str]] = []
    if isinstance(target, str) and target in clean_df.columns:
        if clean_df[target].nunique() <= 50:
            for col in cat_cols:
                if col == target:
                    continue
                cross = clean_df.groupby([col, target]).size().reset_index(name="count")
                fig = px.bar(cross, x=col, y="count", color=target, barmode="stack",
                             title=f"Stacked Bar: {col} vs {target}")
                fig.update_layout(xaxis_tickangle=-45)
                figs.append((fig, f"stacked__{col}__vs__{target}.html"))
    return figs


def plot_scatter_with_trendline(clean_df: pd.DataFrame, num_cols: List[str], max_pairs: int = 2) -> List[Tuple[go.Figure, str]]:
    figs: List[Tuple[go.Figure, str]] = []
    if len(num_cols) >= 2:
        x_col = num_cols[0]
        for y_col in num_cols[1:1 + max_pairs]:
            try:
                fig = px.scatter(clean_df, x=x_col, y=y_col, trendline="ols",
                                 title=f"Scatter with Trendline: {x_col} vs {y_col}")
            except Exception:
                fig = px.scatter(clean_df, x=x_col, y=y_col,
                                 title=f"Scatter (trendline unavailable): {x_col} vs {y_col}")
            figs.append((fig, f"scatter__{x_col}__vs__{y_col}.html"))
    return figs


def plot_violin_target_by_cat(clean_df: pd.DataFrame, target: Optional[str], cat_cols: List[str], max_cols: int = 3) -> List[Tuple[go.Figure, str]]:
    figs: List[Tuple[go.Figure, str]] = []
    if isinstance(target, str) and target in clean_df.columns and pd.api.types.is_numeric_dtype(clean_df[target]):
        for col in cat_cols[:max_cols]:
            fig = px.violin(clean_df, x=col, y=target, box=True, points="all",
                            title=f"Violin: {target} by {col}")
            figs.append((fig, f"violin__{target}__by__{col}.html"))
    return figs


def plot_strip_numeric_by_target(clean_df: pd.DataFrame, target: Optional[str], num_cols: List[str], max_cols: int = 5) -> List[Tuple[go.Figure, str]]:
    figs: List[Tuple[go.Figure, str]] = []
    if isinstance(target, str) and target in clean_df.columns and not pd.api.types.is_numeric_dtype(clean_df[target]):
        for col in num_cols[:max_cols]:
            fig = px.strip(clean_df, x=target, y=col, title=f"Strip: {col} by {target}")
            figs.append((fig, f"strip__{col}__by__{target}.html"))
    return figs


def load_and_prepare(csv_path: Path, seed: int, max_cat_cardinality: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    set_seeds(seed)
    df = pd.read_csv(csv_path)
    numeric_cols, categorical_cols, datetime_cols = infer_types(df, max_cat_cardinality)
    clean_df = clean_dataframe(df, numeric_cols, categorical_cols, datetime_cols)
    return df, clean_df, numeric_cols, categorical_cols, datetime_cols


def save_figs(figs: Iterable[Tuple[go.Figure, str]], outdir: Path) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for fig, name in figs:
        path = outdir / name
        fig.write_html(str(path))
        paths.append(path)
    return paths


# Registry of graph generators
GRAPH_REGISTRY = {
    "missing_counts": lambda df, clean_df, num_cols, cat_cols, dt_cols, cfg: [plot_missing_counts(df)],
    "missing_heatmap": lambda df, clean_df, num_cols, cat_cols, dt_cols, cfg: [plot_missing_heatmap(df, cfg.get("seed", 42))],
    "numeric_hists": lambda df, clean_df, num_cols, cat_cols, dt_cols, cfg: plot_numeric_histograms(clean_df, num_cols, cfg.get("n_bins", 30), cfg.get("max_cols", 20)),
    "numeric_boxes": lambda df, clean_df, num_cols, cat_cols, dt_cols, cfg: plot_numeric_boxplots(clean_df, num_cols, cfg.get("max_cols", 20)),
    "corr_pearson": lambda df, clean_df, num_cols, cat_cols, dt_cols, cfg: [plot_correlations(clean_df, num_cols)[0]] if plot_correlations(clean_df, num_cols) else [],
    "corr_spearman": lambda df, clean_df, num_cols, cat_cols, dt_cols, cfg: [plot_correlations(clean_df, num_cols)[1]] if len(plot_correlations(clean_df, num_cols)) > 1 else [],
    "cat_bars": lambda df, clean_df, num_cols, cat_cols, dt_cols, cfg: plot_categorical_bars(clean_df, cat_cols, cfg.get("max_cols", 20)),
    "stacked_bars": lambda df, clean_df, num_cols, cat_cols, dt_cols, cfg: plot_stacked_vs_target(clean_df, cat_cols, cfg.get("target")),
    "scatter": lambda df, clean_df, num_cols, cat_cols, dt_cols, cfg: plot_scatter_with_trendline(clean_df, num_cols, cfg.get("max_pairs", 2)),
    "violin": lambda df, clean_df, num_cols, cat_cols, dt_cols, cfg: plot_violin_target_by_cat(clean_df, cfg.get("target"), cat_cols, cfg.get("max_cols", 3)),
    "strip": lambda df, clean_df, num_cols, cat_cols, dt_cols, cfg: plot_strip_numeric_by_target(clean_df, cfg.get("target"), num_cols, cfg.get("max_cols", 5)),
}


def generate_graphs(csv_path: Path,
                    outdir: Path,
                    graph_key: str,
                    target: Optional[str] = None,
                    seed: int = 42,
                    max_cat_cardinality: int = 50,
                    n_bins: int = 30,
                    max_cols: int = 20,
                    max_pairs: int = 2) -> List[Path]:
    df, clean_df, num_cols, cat_cols, dt_cols = load_and_prepare(csv_path, seed, max_cat_cardinality)
    cfg = {
        "target": target,
        "seed": seed,
        "n_bins": n_bins,
        "max_cols": max_cols,
        "max_pairs": max_pairs,
    }
    if graph_key == "all":
        order = [
            "missing_counts", "missing_heatmap",
            "numeric_hists", "numeric_boxes",
            "corr_pearson", "corr_spearman",
            "cat_bars", "stacked_bars",
            "scatter", "violin", "strip",
        ]
        all_paths: List[Path] = []
        for key in order:
            figs = GRAPH_REGISTRY[key](df, clean_df, num_cols, cat_cols, dt_cols, cfg)
            all_paths.extend(save_figs(figs, outdir))
        return all_paths
    else:
        if graph_key not in GRAPH_REGISTRY:
            raise ValueError(f"Unknown graph_key '{graph_key}'. Valid: {['all'] + list(GRAPH_REGISTRY.keys())}")
        figs = GRAPH_REGISTRY[graph_key](df, clean_df, num_cols, cat_cols, dt_cols, cfg)
        return save_figs(figs, outdir)




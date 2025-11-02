"""
Autism Screening - Reproducible Analysis and Baselines

This script is written as a notebook-style, self-contained pipeline that you can
run as a Python script or copy into a Jupyter notebook. It follows the spec:

0) Setup & Reproducibility
1) Load Data (from csv_text string)
2) Clean & Encode
3) EDA & Visuals
4) Baseline & Metrics Helpers
5) Models: (A) PyTorch Logistic Regression; (B) RandomForest
6) Optional Explainability (SHAP)
7) Clustering
8) Fairness & Leakage Checks
9) Final Comparison & Summary Table
10) Slides command (Reveal.js) + Executive Summary
11) Nice-to-Haves (some implemented where feasible)
12) Deliverables: cleaned CSV, figures/, slides command

Important: This is a statistical classification exercise on self-reported
screening data. It is not medical advice or diagnosis. Use neutral language
and review limitations.
"""

from __future__ import annotations

import os
import sys
import json
import math
import textwrap
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional deps
try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    from sklearn.manifold import TSNE  # fallback
    HAS_UMAP = False

try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# 0) Setup & Reproducibility ----------------------------------------------------

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

plt.style.use("seaborn-v0_8-whitegrid")

FIG_DIR = "figures"
DATA_DIR = "data"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def savefig(name: str) -> None:
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.show()


# 1) Load Data ------------------------------------------------------------------

# Paste your CSV into csv_text below (keep triple quotes)
csv_text = """
"""

def load_from_text(csv_text: str) -> pd.DataFrame:
    from io import StringIO
    df = pd.read_csv(StringIO(csv_text), na_values=["?", ""], keep_default_na=True)
    return df


def load_from_path(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}")
    df = pd.read_csv(path, na_values=["?", ""], keep_default_na=True)
    return df


def to_snake(name: str) -> str:
    name = name.strip().replace("/", "_").replace(" ", "_")
    # Handle common known typo
    name = name.replace("contry_of_res", "country_of_res")
    return name.lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake(c) for c in df.columns]
    return df


def basic_overview(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("Head:\n", df.head())
    print("\nMissing per column:\n", df.isna().sum())
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    print("\nCategorical unique values (first 15):")
    for c in cat_cols:
        vals = df[c].dropna().unique()[:15]
        print(f"- {c}: {vals}")


# 2) Clean & Encode --------------------------------------------------------------

def clean_and_encode(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    df = df_raw.copy()

    # Strip whitespace, lowercase strings
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.lower().replace({"nan": np.nan})

    # Map booleans
    yes_no_map = {"yes": 1, "no": 0}
    for col in ["jundice", "austim", "used_app_before", "class_asd"]:
        if col in df.columns:
            df[col] = df[col].map(yes_no_map).astype("float")

    # Gender map m->1, f->0, else NaN
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"m": 1, "f": 0})

    # Identify numeric candidates (including A1..A10, age, result, binary maps)
    aq_cols = [f"a{i}_score" for i in range(1, 11) if f"a{i}_score" in df.columns]
    numeric_cols = list({
        *aq_cols,
        *(c for c in ["age", "result", "jundice", "austim", "used_app_before", "gender"] if c in df.columns),
    })

    # One-hot columns
    onehot_cols = [c for c in ["ethnicity", "country_of_res", "age_desc", "relation"] if c in df.columns]

    # Any remaining object/string columns (excluding the above and target)
    obj_cols = [c for c in df.columns if df[c].dtype == object and c not in onehot_cols and c != "class_asd"]
    onehot_cols += obj_cols

    # Impute numeric with median
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            med = df[c].median()
            df[c] = df[c].fillna(med)

    # One-hot encode (drop_first=True to limit collinearity)
    df_dum = pd.get_dummies(df, columns=onehot_cols, drop_first=True, dummy_na=False)

    # Target y
    if "class_asd" not in df_dum.columns:
        raise ValueError("Target column 'class_asd' missing after cleaning. Check CSV content.")
    y = df_dum["class_asd"].astype(int)

    # Feature matrix X: pick defined predictors + any dummies
    base_feats = list({
        *aq_cols,
        *(c for c in ["age", "result", "jundice", "austim", "used_app_before", "gender"] if c in df_dum.columns),
    })
    # Add all dummy columns except the target
    dummy_feats = [c for c in df_dum.columns if c not in base_feats and c != "class_asd"]
    features = base_feats + dummy_feats
    X = df_dum[features].copy()

    return X, y, df_dum, features


# 3) EDA & Visuals ---------------------------------------------------------------

def plot_class_balance(y: pd.Series) -> None:
    plt.figure(figsize=(5, 4))
    sns.countplot(x=y)
    plt.title("Class Balance (class_asd)")
    plt.xlabel("class_asd")
    plt.ylabel("count")
    savefig("eda_class_balance")


def plot_corr_heatmap(df_num: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    corr = df_num.corr(numeric_only=True)
    sns.heatmap(corr, cmap="vlag", annot=True, fmt=".2f", cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap (numeric)")
    savefig("eda_corr_heatmap")


def plot_hists_by_class(df: pd.DataFrame, y: pd.Series, cols: List[str]) -> None:
    for col in cols:
        if col in df.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(data=df.assign(class_asd=y.values), x=col, hue="class_asd", bins=30, stat="count", kde=False)
            plt.title(f"Distribution of {col} by Class")
            savefig(f"hist_{col}_by_class")


def plot_cat_counts_by_class(df_raw: pd.DataFrame, y: pd.Series, cats: List[str]) -> None:
    for col in cats:
        if col in df_raw.columns:
            plt.figure(figsize=(7, 4))
            sns.countplot(data=df_raw.assign(class_asd=y.values), x=col, hue="class_asd")
            plt.title(f"Countplot of {col} by Class")
            plt.xticks(rotation=30, ha="right")
            savefig(f"count_{col}_by_class")


def pca_views(X: pd.DataFrame, y: pd.Series, n_components: int = 3) -> Tuple[np.ndarray, PCA]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=SEED)
    pcs = pca.fit_transform(Xs)

    # 2D
    plt.figure(figsize=(6, 5))
    plt.scatter(pcs[:, 0], pcs[:, 1], c=y.values, cmap="coolwarm", s=18, alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D (PC1 vs PC2)")
    savefig("pca_2d")

    # 3D (matplotlib)
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], c=y.values, cmap="coolwarm", s=18, alpha=0.8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA 3D")
        plt.colorbar(sc, ax=ax, shrink=0.6)
        savefig("pca_3d")
    except Exception:
        pass

    return pcs, pca


def manifold_view(X: pd.DataFrame, y: pd.Series) -> None:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # Plots removed per request; keep function stub (no-op)
    return None


# 4) Baseline & Metrics Helpers --------------------------------------------------

def evaluate_clf(model, X_train, y_train, X_test, y_test, name: str, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    # Predict probabilities if available
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        if y_prob.ndim == 2 and y_prob.shape[1] > 1:
            y_prob_1 = y_prob[:, 1]
        else:
            y_prob_1 = y_prob.ravel()
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # Convert scores to [0,1] via logistic for binary
        y_prob_1 = 1 / (1 + np.exp(-scores))
    else:
        # As a fallback, use predictions as probabilities (not ideal)
        y_prob_1 = model.predict(X_test)

    y_pred = (y_prob_1 >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        roc_auc = float(roc_auc_score(y_test, y_prob_1))
    except Exception:
        roc_auc = float("nan")
    pr_auc = float(average_precision_score(y_test, y_prob_1))

    # Skip ROC/PR curve plots per request (for all models)
    do_curves = False
    if do_curves:
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob_1)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC - {name}")
        plt.legend()
        savefig(f"roc_{name}")

        # PR curve
        pr, rc, _ = precision_recall_curve(y_test, y_prob_1)
        plt.figure(figsize=(5, 4))
        plt.plot(rc, pr, label=f"PR AUC={pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR - {name}")
        plt.legend()
        savefig(f"pr_{name}")

    # Confusion matrix (skip for RandomForest per request)
    if name.lower() != "randomforest":
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {name}")
        savefig(f"cm_{name}")

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4) if not math.isnan(roc_auc) else None,
        "PR_AUC": round(pr_auc, 4),
    }


def plot_feature_importance(model, feature_names: List[str], top_n: int = 20, name: str = "model") -> None:
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        coef = coef.ravel() if coef.ndim > 1 else coef
        imp = np.abs(coef)
    if imp is None:
        print("Feature importance not available for this model.")
        return
    order = np.argsort(imp)[-top_n:]
    names = [feature_names[i] for i in order]
    vals = imp[order]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=vals, y=names, orient="h")
    plt.title(f"Top-{top_n} Feature Importances - {name}")
    savefig(f"featimp_{name}")


# 5A) PyTorch Logistic Regression ----------------------------------------------

class TorchLogReg(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x)


def train_torch_logreg(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train/val split from train
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_s, y_train.values.astype(np.int64), test_size=0.2, random_state=SEED, stratify=y_train
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TorchLogReg(in_dim=X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    X_tr_t, y_tr_t = to_tensor(X_tr), to_tensor(y_tr).view(-1, 1)
    X_val_t, y_val_t = to_tensor(X_val), to_tensor(y_val).view(-1, 1)

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")
    patience = 5
    wait = 0

    max_epochs = 200
    batch_size = 128
    for epoch in range(1, max_epochs + 1):
        # Mini-batch training
        model.train()
        perm = np.random.permutation(len(X_tr_t))
        batch_losses = []
        for i in range(0, len(X_tr_t), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        train_loss = float(np.mean(batch_losses))

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Plot training curve
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("TorchLogReg Training Curve")
    plt.legend()
    savefig("torch_logreg_training_curve")

    # Evaluate on test
    X_te_t = to_tensor(X_test_s)
    model.eval()
    with torch.no_grad():
        logits = model(X_te_t).cpu().numpy().ravel()
    probs = 1 / (1 + np.exp(-logits))

    class SKLike:
        def predict_proba(self, X):
            return np.c_[1 - probs, probs]

    metrics = evaluate_clf(SKLike(), X_train, y_train, X_test, y_test, name="TorchLogReg")
    return metrics, history


# 5B) RandomForest ---------------------------------------------------------------

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, feature_names: List[str]) -> Tuple[Dict[str, float], RandomForestClassifier]:
    rf = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)
    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 3],
    }
    gs = GridSearchCV(rf, param_grid=param_grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    print("RF best params:", gs.best_params_)
    metrics = evaluate_clf(best, X_train, y_train, X_test, y_test, name="RandomForest")
    plot_feature_importance(best, feature_names, top_n=20, name="RandomForest")
    return metrics, best


# 6) Optional Explainability -----------------------------------------------------

def explain_with_shap(model, X_train: pd.DataFrame, feature_names: List[str], sample_n: int = 500) -> None:
    if not HAS_SHAP:
        print("SHAP not available; skipping.")
        return
    try:
        X_sample = X_train.sample(min(sample_n, len(X_train)), random_state=SEED)
        shap_available = False
        if hasattr(model, "estimators_") or hasattr(model, "tree_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            shap_available = True
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_sample, link="logit")
            shap_values = explainer.shap_values(X_sample)
            shap_available = True
        if shap_available:
            plt.figure(figsize=(7, 5))
            try:
                shap.summary_plot(shap_values if isinstance(shap_values, np.ndarray) else shap_values[1], X_sample, feature_names=feature_names, show=False)
                savefig("shap_summary")
            except Exception:
                pass
            try:
                shap.summary_plot(shap_values if isinstance(shap_values, np.ndarray) else shap_values[1], X_sample, feature_names=feature_names, plot_type="bar", show=False)
                savefig("shap_bar")
            except Exception:
                pass
    except Exception as e:
        print("SHAP explanation failed:", str(e))


# 7) Clustering ------------------------------------------------------------------

def clustering_analysis(X: pd.DataFrame, y: pd.Series, pcs: Optional[np.ndarray]) -> None:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca2 = PCA(n_components=10, random_state=SEED)
    Xp = pca2.fit_transform(Xs)

    sil_scores = {}
    try:
        from sklearn.metrics import silhouette_score
        for k in [2, 3, 4, 5, 6]:
            km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
            labs = km.fit_predict(Xp)
            sil = float(silhouette_score(Xp, labs))
            sil_scores[k] = sil
        print("Silhouette scores:", sil_scores)
    except Exception:
        print("silhouette_score unavailable; skipping.")

    # k=2 comparison
    km2 = KMeans(n_clusters=2, random_state=SEED, n_init=10)
    cl2 = km2.fit_predict(Xp)
    print({"ARI": round(float(ARI(y, cl2)), 4), "NMI": round(float(NMI(y, cl2)), 4)})

    # Plots removed per request
    return None


# 8) Fairness & Leakage Checks ---------------------------------------------------

def fairness_check_rf(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    sens_cols = [c for c in X_train.columns if c.startswith("gender_") or c.startswith("ethnicity_") or c.startswith("country_of_res_") or c == "gender"]

    rf_full = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf_full.fit(X_train, y_train)
    if hasattr(rf_full, "predict_proba"):
        y_prob_full = rf_full.predict_proba(X_test)[:, 1]
    else:
        y_prob_full = rf_full.predict(X_test)
    auc_full = roc_auc_score(y_test, y_prob_full)

    X_train_drop = X_train.drop(columns=[c for c in sens_cols if c in X_train.columns], errors="ignore")
    X_test_drop = X_test.drop(columns=[c for c in sens_cols if c in X_test.columns], errors="ignore")
    rf_drop = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf_drop.fit(X_train_drop, y_train)
    if hasattr(rf_drop, "predict_proba"):
        y_prob_drop = rf_drop.predict_proba(X_test_drop)[:, 1]
    else:
        y_prob_drop = rf_drop.predict(X_test_drop)
    auc_drop = roc_auc_score(y_test, y_prob_drop)
    print({"AUC_full": round(float(auc_full), 4), "AUC_without_sensitive": round(float(auc_drop), 4), "AUC_drop": round(float(auc_full - auc_drop), 4)})


# 9) Final Comparison & Summary Table -------------------------------------------

def summarize_results(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    # Choose best by ROC_AUC then F1
    best = df.sort_values(by=["ROC_AUC", "F1"], ascending=False).head(1)
    print("\nModel comparison:\n", df)
    print("\nBest model (by ROC_AUC then F1):\n", best)
    return df


def build_html_report(
    html_path: str,
    results: Dict[str, Dict[str, float]],
    summary_df: pd.DataFrame,
    used_umap: bool,
    fig_dir: str = FIG_DIR,
) -> None:
    """Create a simple, useful HTML report with titles and brief captions for each figure."""
    # Ordered list of figures with titles and short captions
    candidates = [
        ("eda_class_balance", "Class Balance", "Counts of class_asd = 0 vs 1."),
        ("eda_corr_heatmap", "Correlation Heatmap", "Correlations across numeric features (darker = stronger)."),
        ("hist_age_by_class", "Age by Class", "Distribution of age split by class labels."),
        ("hist_result_by_class", "Result by Class", "Distribution of result scores by class."),
        ("count_used_app_before_by_class", "Used App Before by Class", "Counts of prior app usage split by class."),
        ("count_austim_by_class", "Austim Flag by Class", "Counts of reported autism flag by class."),
        ("count_jundice_by_class", "Jaundice Flag by Class", "Counts of reported jaundice by class."),
        ("torch_logreg_training_curve", "Torch LogReg Training", "Training/validation loss over epochs."),
        ("cm_TorchLogReg", "Confusion Matrix - TorchLogReg", "Counts of predictions vs truth."),
        ("featimp_RandomForest", "Top Feature Importances", "RandomForest importance for top features."),
        ("shap_summary", "SHAP Summary", "Feature impact summary (if available)."),
        ("shap_bar", "SHAP Mean |SHAP|", "Mean absolute SHAP values (if available)."),
    ]

    # Build HTML
    rows = []
    rows.append("<h1>Autism Screening - Analysis Report</h1>")
    rows.append("<p>This is a concise, neutral analysis of a self-reported screening dataset. It is not medical advice.</p>")

    # Summary metrics table
    try:
        rows.append("<h2>Model Summary</h2>")
        rows.append(summary_df.to_html(border=1, classes="table", float_format=lambda x: f"{x:.4f}"))
    except Exception:
        pass

    # Individual figures
    for fname, title, caption in candidates:
        path = os.path.join(fig_dir, f"{fname}.png")
        if os.path.exists(path):
            rows.append(f"<h3>{title}</h3>")
            rows.append(f"<p>{caption}</p>")
            rows.append(f"<img src='{path}' alt='{title}' style='max-width: 900px; width: 100%; height: auto;' />")

    # Executive summary
    rows.append("<h2>Executive Summary</h2>")
    best_name = max(results.keys(), key=lambda k: (results[k]["ROC_AUC"] if results[k]["ROC_AUC"] is not None else -1, results[k]["F1"]))
    best_metrics = results[best_name]
    rows.append("<ul>")
    rows.append(f"<li><b>Best model:</b> {best_name} | {json.dumps(best_metrics)}</li>")
    rows.append("<li><b>Notes:</b> cleaned columns, encoded categoricals, scaled numerics, stratified splits.</li>")
    rows.append("<li><b>Fairness check:</b> compared AUC with and without sensitive columns.</li>")
    rows.append("<li><b>Limitations:</b> label noise, sampling bias, sensitive attributes, small sample size.</li>")
    rows.append("</ul>")

    html = f"""
    <html>
    <head>
      <meta charset='utf-8'>
      <title>Autism Screening - Report</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
        h1, h2, h3 {{ color: #222; }}
        p {{ color: #333; }}
        .table {{ border-collapse: collapse; }}
        .table th, .table td {{ padding: 6px 10px; }}
      </style>
    </head>
    <body>
      {''.join(rows)}
    </body>
    </html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("Saved HTML report to:", html_path)


def main() -> None:
    # Load
    if csv_text.strip():
        df_raw = load_from_text(csv_text)
    else:
        default_path = os.path.join(os.path.dirname(__file__), "autism_screening.csv")
        try:
            df_raw = load_from_path(default_path)
            print(f"Loaded CSV from file: {default_path}")
        except FileNotFoundError:
            print("No csv_text provided and default file not found at ./autism_screening.csv.")
            print("Please paste into csv_text or place the file next to final.py and rerun.")
            return
    df_raw = standardize_columns(df_raw)

    # Print overview
    basic_overview(df_raw)

    # Clean & encode
    X, y, df_dum, feature_names = clean_and_encode(df_raw)

    # Save cleaned
    clean_path = os.path.join(DATA_DIR, "clean_autism.csv")
    df_dum.to_csv(clean_path, index=False)
    print("Saved cleaned dataset to:", clean_path)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )

    # EDA
    plot_class_balance(y)
    # Correlation heatmap for numeric only
    plot_corr_heatmap(df_dum.select_dtypes(include=[np.number]))
    plot_hists_by_class(df_dum, y, cols=[c for c in ["age", "result"] if c in df_dum.columns])
    plot_cat_counts_by_class(df_raw, y, cats=[c for c in ["used_app_before", "austim", "jundice"] if c in df_raw.columns])
    # PCA and manifold plots removed per request
    pcs = None

    # Pair/2D view removed per request

    # 5A) Torch logistic regression
    torch_metrics, _ = train_torch_logreg(X_train, y_train, X_test, y_test)
    print("TorchLogReg metrics:", torch_metrics)

    # 5B) Random Forest
    rf_metrics, rf_model = train_random_forest(X_train, y_train, X_test, y_test, feature_names)
    print("RandomForest metrics:", rf_metrics)

    # 6) Explainability
    explain_with_shap(rf_model, X_train, feature_names)

    # 7) Clustering (scores only; plots removed)
    clustering_analysis(X, y, pcs=None)

    # 8) Fairness
    fairness_check_rf(X_train, y_train, X_test, y_test)

    # 9) Summary
    results = {
        "TorchLogReg": torch_metrics,
        "RandomForest": rf_metrics,
    }
    summary_df = summarize_results(results)
    summary_path = os.path.join(DATA_DIR, "model_summary.csv")
    summary_df.to_csv(summary_path)
    print("Saved summary to:", summary_path)

    # HTML report
    html_path = os.path.join(os.path.dirname(__file__), "final.html")
    # Remove stale images to ensure they don't appear on live servers
    for stale in [
        os.path.join(FIG_DIR, "roc_RandomForest.png"),
        os.path.join(FIG_DIR, "pr_RandomForest.png"),
        os.path.join(FIG_DIR, "roc_TorchLogReg.png"),
        os.path.join(FIG_DIR, "pr_TorchLogReg.png"),
        os.path.join(FIG_DIR, "cm_RandomForest.png"),
        os.path.join(FIG_DIR, "pca2d_class.png"),
        os.path.join(FIG_DIR, "pca2d_cluster_k2.png"),
        os.path.join(FIG_DIR, "pca_2d.png"),
        os.path.join(FIG_DIR, "pca_3d.png"),
        os.path.join(FIG_DIR, "umap_2d.png"),
        os.path.join(FIG_DIR, "tsne_2d.png"),
        os.path.join(FIG_DIR, "pair_2d_top2.png"),
    ]:
        try:
            if os.path.exists(stale):
                os.remove(stale)
        except Exception:
            pass
    build_html_report(
        html_path=html_path,
        results=results,
        summary_df=summary_df,
        used_umap=HAS_UMAP,
        fig_dir=FIG_DIR,
    )

    # 10) Slides command
    print("\nTo export slides from a notebook version, run (after converting to index.ipynb):")
    print("!jupyter nbconvert --to slides --TemplateExporter.exclude_input=True --no-prompt --reveal-prefix https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.5.0 index.ipynb --output slides.html")

    # Executive Summary (printed)
    print("\nExecutive Summary (short):")
    print("- Dataset size:", df_raw.shape)
    print("- Cleaning: lower-snake-case, YES/NO->1/0, gender m=1 f=0, dummies, median impute.")
    try:
        print("- Top numeric signals:", df_dum.corr(numeric_only=True)["class_asd"].drop("class_asd").abs().sort_values(ascending=False).head(5).to_dict())
    except Exception:
        pass
    best_name = max(results.keys(), key=lambda k: (results[k]["ROC_AUC"] if results[k]["ROC_AUC"] is not None else -1, results[k]["F1"]))
    print(f"- Best model: {best_name} | Metrics: {results[best_name]}")
    print("- Fairness: compared ROC-AUC with/without sensitive columns; review AUC_drop above.")
    print("- Limitations: label noise, sampling bias, sensitive attributes, small sample size.")


if __name__ == "__main__":
    main()



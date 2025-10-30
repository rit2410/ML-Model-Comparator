# File: src/evaluate.py
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import set_plot_theme


set_plot_theme()


def _get_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            return proba
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_test)
            # Convert to probabilities-like scores via softmax for multi-class visual AUC
            if scores.ndim == 1:
                scores = scores.reshape(-1, 1)
            return scores
        except Exception:
            pass
    return None


def _plot_confusion(cm: np.ndarray, class_names: List[str], title: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names, rotation=0)
    fig.tight_layout()
    return fig


def _plot_roc_ovr(y_true: np.ndarray, scores: np.ndarray, class_names: List[str]):
    n_classes = len(class_names)
    y_true_bin = np.eye(n_classes)[y_true]

    fig, ax = plt.subplots(figsize=(5, 4))
    try:
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], scores[:, i])
            ax.plot(fpr, tpr, label=f"{class_names[i]}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC (One-vs-Rest)")
        ax.legend()
        fig.tight_layout()
        return fig
    except Exception:
        plt.close(fig)
        return None


def evaluate_model(
    model_name: str,
    model,
    X_test,
    y_test,
    class_names: List[str],
) -> Dict:
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = _plot_confusion(cm, class_names, f"Confusion Matrix – {model_name}")

    scores = _get_scores(model, X_test)
    roc_fig = None
    if scores is not None:
        # Ensure 2D scores with right shape
        if scores.ndim == 1:
            scores = np.vstack([1 - scores, scores]).T
        if scores.shape[1] != len(class_names):
            # best-effort pad/trim
            if scores.shape[1] == 1 and len(class_names) == 2:
                scores = np.hstack([1 - scores, scores])
            else:
                scores = None
        if scores is not None:
            roc_fig = _plot_roc_ovr(y_test, scores, class_names)

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }

    return {
        "model_name": model_name,
        "metrics": metrics,
        "report": report,
        "confusion_fig": cm_fig,
        "roc_fig": roc_fig,
    }


def summarize_results_table(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append(
            {
                "Model": r["model_name"],
                "Accuracy": round(m["accuracy"], 4),
                "Precision (macro)": round(m["precision_macro"], 4),
                "Recall (macro)": round(m["recall_macro"], 4),
                "F1 (macro)": round(m["f1_macro"], 4),
            }
        )
    return pd.DataFrame(rows).sort_values(by=["F1 (macro)", "Accuracy"], ascending=False).reset_index(drop=True)

"""Evaluation metrics and the shared results file.

All methods output p = probability that the image is FAKE. We threshold at 0.5
for accuracy / precision / recall / F1, and use the raw p for ROC-AUC.
Every evaluation run appends one row to results/results.csv so the tables in
the report can be rebuilt at any time with scripts/make_tables.py.
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

FIELDS = ["time", "method", "train_set", "test_set", "perturbation", "n",
          "accuracy", "precision", "recall", "f1", "auc", "ap", "real_acc", "fake_acc", "tn", "fp", "fn", "tp"]


def compute_metrics(y_true, p_fake, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p_fake = np.asarray(p_fake, dtype=float)
    y_pred = (p_fake >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, p_fake) if both else float("nan"),
        "ap": average_precision_score(y_true, p_fake) if both else float("nan"),
        # accuracy on each class separately: tells you if a model just says "fake" for everything
        "real_acc": tn / (tn + fp) if (tn + fp) else float("nan"),
        "fake_acc": tp / (tp + fn) if (tp + fn) else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def append_result(csv_path: str | Path, row: dict) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    row = {"time": dt.datetime.now().isoformat(timespec="seconds"), **row}
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})


def pretty(m: dict) -> str:
    return (f"acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}  "
            f"real_acc={m['real_acc']:.4f}  fake_acc={m['fake_acc']:.4f}  (n={m['n']})")

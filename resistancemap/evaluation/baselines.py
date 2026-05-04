"""Required baseline panel.

Every ResistanceMap metric reported publicly must appear in a table with ALL
baselines computed on the IDENTICAL split. A 0.88 -> 0.89 delta over
``per_drug_mean`` is not publishable; a delta over ``deepcdr`` on a
drug-holdout split is.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression


# ---------------------------------------------------------------------------
# Regression baselines
# ---------------------------------------------------------------------------
def predict_global_mean(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        target: str) -> np.ndarray:
    mu = float(train_df[target].mean())
    return np.full(len(test_df), mu)


def predict_per_drug_mean(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          target: str) -> np.ndarray:
    """Per-drug mean; falls back to global mean for unseen drugs (drug_holdout)."""
    global_mu = float(train_df[target].mean())
    means = train_df.groupby("drug")[target].mean()
    return test_df["drug"].map(means).fillna(global_mu).to_numpy()


def ridge_on_esm2_meanpool(
    train_emb: np.ndarray, train_y: np.ndarray, test_emb: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """Ridge on mean-pooled ESM-2 embeddings. Often matches fancy GNNs at n~150."""
    return Ridge(alpha=alpha).fit(train_emb, train_y).predict(test_emb)


# ---------------------------------------------------------------------------
# Classification baselines
# ---------------------------------------------------------------------------
def predict_base_rate(train_y: np.ndarray, n_test: int) -> np.ndarray:
    p = float(np.asarray(train_y).mean())
    return np.full(n_test, p)


def logistic_on_esm2(
    train_emb: np.ndarray, train_y: np.ndarray, test_emb: np.ndarray
) -> np.ndarray:
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(train_emb, train_y)
    return clf.predict_proba(test_emb)[:, 1]


# ---------------------------------------------------------------------------
# External SOTA reimpl -- stub; plug in DeepCDR / PaccMann reimpls
# ---------------------------------------------------------------------------
def deepcdr_reimpl(train_df, test_df):
    raise NotImplementedError(
        "Plug in DeepCDR reimpl (or published weights) on IDENTICAL splits. "
        "Do not report headline numbers without a published-SOTA row."
    )


BASELINES_REGRESSION: dict[str, Callable] = {
    "global_mean":    predict_global_mean,
    "per_drug_mean":  predict_per_drug_mean,
    "ridge_esm2":     ridge_on_esm2_meanpool,
    "deepcdr":        deepcdr_reimpl,
}

BASELINES_CLASSIFICATION: dict[str, Callable] = {
    "base_rate":      predict_base_rate,
    "logistic_esm2":  logistic_on_esm2,
    "deepcdr":        deepcdr_reimpl,
}


def require_baseline_row(results_table: pd.DataFrame) -> None:
    """Fail loudly if the results table is missing required baseline rows."""
    required = {"per_drug_mean", "ridge_esm2"}
    have = set(results_table["model"]) if "model" in results_table.columns else set()
    missing = required - have
    if missing:
        raise AssertionError(
            f"Missing required baselines: {missing}. "
            f"No headline publishable without them on the same split."
        )

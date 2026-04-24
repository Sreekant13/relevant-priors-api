import math
import re
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _clean_series(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _modality_one(x: str) -> str:
    x = (x or "").strip().upper()
    if x.startswith(("MRI", "MR ")):
        return "MR"
    if x.startswith(("CT", "CTA")):
        return "CT"
    if x.startswith(("XR", "X RAY", "XRAY", "DX")):
        return "XR"
    if x.startswith("US") or "ULTRASOUND" in x:
        return "US"
    if x.startswith("NM") or "SPECT" in x:
        return "NM"
    if x.startswith("PET"):
        return "PET"
    if x.startswith("MG") or "MAMMO" in x:
        return "MG"
    if x.startswith("FL"):
        return "FL"
    parts = x.split()
    return parts[0] if parts else "UNK"


def _get_modality(s: pd.Series) -> np.ndarray:
    return np.array([_modality_one(x) for x in s])


def _jaccard(a: pd.Series, b: pd.Series) -> np.ndarray:
    out = []
    for x, y in zip(a, b):
        xs, ys = set(x.split()), set(y.split())
        out.append(len(xs & ys) / max(1, len(xs | ys)))
    return np.array(out, dtype=float)


def _intersection_count(a: pd.Series, b: pd.Series) -> np.ndarray:
    out = []
    for x, y in zip(a, b):
        out.append(len(set(x.split()) & set(y.split())))
    return np.array(out, dtype=float)


def _days_between(current_dates: pd.Series, prior_dates: pd.Series) -> np.ndarray:
    current = pd.to_datetime(current_dates, errors="coerce")
    prior = pd.to_datetime(prior_dates, errors="coerce")
    days = (current - prior).dt.days.fillna(9999).clip(lower=0).to_numpy()
    return days.astype(float)


def build_feature_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["current_desc", "prior_desc", "current_date", "prior_date"])


def pair_text(X: pd.DataFrame) -> List[str]:
    return (X["current_desc"].fillna("") + " [SEP] " + X["prior_desc"].fillna("")).tolist()


def engineered_features(X: pd.DataFrame) -> np.ndarray:
    current = _clean_series(X["current_desc"])
    prior = _clean_series(X["prior_desc"])
    days = _days_between(X["current_date"], X["prior_date"])

    features = np.vstack(
        [
            (current == prior).astype(float).to_numpy(),
            (_get_modality(current) == _get_modality(prior)).astype(float),
            _jaccard(current, prior),
            _intersection_count(current, prior),
            days / 3650.0,
            np.log1p(days) / 10.0,
            (days <= 365).astype(float),
            (days <= 730).astype(float),
            (days <= 1095).astype(float),
        ]
    ).T
    return features.astype(float)


class RelevantPriorsModel:
    """Small, deterministic model for deciding whether prior radiology exams are relevant."""

    def __init__(self, threshold: float = 0.45, max_features: int = 10000, C: float = 1.0):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            max_features=max_features,
            min_df=1,
        )
        self.classifier = LogisticRegression(max_iter=1000, C=C, solver="liblinear")

    def _features(self, X: pd.DataFrame, fit: bool = False):
        texts = pair_text(X)
        if fit:
            text_features = self.vectorizer.fit_transform(texts)
        else:
            text_features = self.vectorizer.transform(texts)
        numeric_features = sparse.csr_matrix(engineered_features(X))
        return sparse.hstack([text_features, numeric_features], format="csr")

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        features = self._features(X, fit=True)
        self.classifier.fit(features, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        features = self._features(X, fit=False)
        return self.classifier.predict_proba(features)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X) >= self.threshold


def cases_to_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for case in payload.get("cases", []):
        current = case.get("current_study", {}) or {}
        current_desc = current.get("study_description", "") or ""
        current_date = current.get("study_date", "") or ""

        for prior in case.get("prior_studies", []) or []:
            rows.append(
                {
                    "case_id": str(case.get("case_id", "")),
                    "study_id": str(prior.get("study_id", "")),
                    "current_desc": current_desc,
                    "prior_desc": prior.get("study_description", "") or "",
                    "current_date": current_date,
                    "prior_date": prior.get("study_date", "") or "",
                }
            )
    return rows

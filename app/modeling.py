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

def _body_region_one(x: str) -> str:
    x = (x or "").strip().upper()

    region_patterns = [
        ("BRAIN_HEAD", ["BRAIN", "HEAD", "SKULL", "ORBITS", "ORBIT", "FACE", "FACIAL", "SINUS", "SINUSES", "IAC"]),
        ("NECK", ["NECK", "SOFT TISSUE NECK", "CERVICAL SOFT TISSUE"]),
        ("CHEST", ["CHEST", "THORAX", "LUNG", "LUNGS", "RIB", "RIBS", "STERNUM"]),
        ("CARDIAC", ["CARDIAC", "HEART", "CORONARY", "CTA CORONARY"]),
        ("ABDOMEN", ["ABDOMEN", "ABD", "LIVER", "KIDNEY", "RENAL", "PANCREAS", "GALLBLADDER", "RUQ"]),
        ("PELVIS", ["PELVIS", "PELVIC", "HIP", "HIPS", "SACRUM", "SI JOINT"]),
        ("ABD_PELVIS", ["ABDOMEN PELVIS", "ABD PELVIS", "A/P", "KUB"]),
        ("SPINE_C", ["C SPINE", "CERVICAL SPINE", "CERVICAL"]),
        ("SPINE_T", ["T SPINE", "THORACIC SPINE", "THORACIC"]),
        ("SPINE_L", ["L SPINE", "LUMBAR SPINE", "LUMBAR"]),
        ("SPINE", ["SPINE", "MYELOGRAM"]),
        ("BREAST", ["BREAST", "MAMMO", "MAMMOGRAM"]),
        ("UPPER_EXT", ["SHOULDER", "HUMERUS", "ELBOW", "FOREARM", "WRIST", "HAND", "FINGER", "ARM"]),
        ("LOWER_EXT", ["FEMUR", "KNEE", "TIBIA", "FIBULA", "ANKLE", "FOOT", "TOE", "LEG"]),
        ("VASCULAR", ["ANGIO", "CTA", "MRA", "ARTERY", "ARTERIES", "VEIN", "VEINS", "VENOUS", "AORTA", "CAROTID"]),
        ("WHOLE_BODY", ["WHOLE BODY", "BONE SCAN", "PET", "METASTATIC"]),
    ]

    for region, keywords in region_patterns:
        for keyword in keywords:
            if keyword in x:
                return region

    return "UNK"


def _get_body_region(s: pd.Series) -> np.ndarray:
    return np.array([_body_region_one(x) for x in s])

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

    current_modality = _get_modality(current)
    prior_modality = _get_modality(prior)

    current_region = _get_body_region(current)
    prior_region = _get_body_region(prior)

    same_modality = (current_modality == prior_modality).astype(float)
    same_region = (current_region == prior_region).astype(float)
    known_same_region = ((current_region != "UNK") & (current_region == prior_region)).astype(float)
    same_modality_and_region = ((current_modality == prior_modality) & (current_region == prior_region)).astype(float)

    features = np.vstack(
        [
            (current == prior).astype(float).to_numpy(),
            same_modality,
            same_region,
            known_same_region,
            same_modality_and_region,
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

        # Word n-grams capture exact phrase relationships such as:
        # "MRI BRAIN", "CT HEAD", "WITHOUT CONTRAST".
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 3),
            max_features=max_features,
            min_df=1,
        )

        # Character n-grams improve robustness to abbreviations and misspellings:
        # "CNTRST" vs "CONTRAST", "WO" vs "WITHOUT", "MR" vs "MRI".
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=max_features,
            min_df=1,
        )

        self.classifier = LogisticRegression(max_iter=1000, C=C, solver="liblinear")

    def _features(self, X: pd.DataFrame, fit: bool = False):
        texts = pair_text(X)

        if fit:
            word_features = self.word_vectorizer.fit_transform(texts)
            char_features = self.char_vectorizer.fit_transform(texts)
        else:
            word_features = self.word_vectorizer.transform(texts)
            char_features = self.char_vectorizer.transform(texts)

        numeric_features = sparse.csr_matrix(engineered_features(X))

        return sparse.hstack(
            [word_features, char_features, numeric_features],
            format="csr",
        )

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

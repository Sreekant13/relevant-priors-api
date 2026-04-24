import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

from app.modeling import RelevantPriorsModel


def load_public_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    truth = {
        (str(item["case_id"]), str(item["study_id"])): int(item["is_relevant_to_current"])
        for item in data["truth"]
    }

    rows = []
    labels = []
    groups = []

    for case in data["cases"]:
        current = case["current_study"]
        case_id = str(case["case_id"])

        for prior in case["prior_studies"]:
            study_id = str(prior["study_id"])

            rows.append(
                {
                    "current_desc": current.get("study_description", "") or "",
                    "prior_desc": prior.get("study_description", "") or "",
                    "current_date": current.get("study_date", "") or "",
                    "prior_date": prior.get("study_date", "") or "",
                }
            )

            labels.append(truth[(case_id, study_id)])
            groups.append(case_id)

    return pd.DataFrame(rows), np.array(labels), np.array(groups)


def main():
    X, y, groups = load_public_json("data/relevant_priors_public.json")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, valid_idx = next(splitter.split(X, y, groups))

    model = RelevantPriorsModel(threshold=0.45)
    model.fit(X.iloc[train_idx], y[train_idx])

    probs = model.predict_proba(X.iloc[valid_idx])
    y_valid = y[valid_idx]

    best = None

    print("Threshold sweep:")
    print("-" * 80)

    for threshold in np.arange(0.20, 0.81, 0.01):
        preds = probs >= threshold

        acc = accuracy_score(y_valid, preds)
        precision = precision_score(y_valid, preds, zero_division=0)
        recall = recall_score(y_valid, preds, zero_division=0)
        f1 = f1_score(y_valid, preds, zero_division=0)
        cm = confusion_matrix(y_valid, preds).tolist()

        if best is None or acc > best["accuracy"]:
            best = {
                "threshold": threshold,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm,
            }

    print("Best threshold:")
    print(f"threshold: {best['threshold']:.2f}")
    print(f"accuracy:  {best['accuracy']:.4f}")
    print(f"precision: {best['precision']:.4f}")
    print(f"recall:    {best['recall']:.4f}")
    print(f"f1:        {best['f1']:.4f}")
    print(f"confusion matrix: {best['confusion_matrix']}")


if __name__ == "__main__":
    main()
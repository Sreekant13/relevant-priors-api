import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression

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


def evaluate_model(model, X_valid, y_valid):
    probs = model.predict_proba(X_valid)

    best = None

    for threshold in np.arange(0.30, 0.701, 0.01):
        preds = probs >= threshold

        acc = accuracy_score(y_valid, preds)
        precision = precision_score(y_valid, preds, zero_division=0)
        recall = recall_score(y_valid, preds, zero_division=0)
        f1 = f1_score(y_valid, preds, zero_division=0)
        cm = confusion_matrix(y_valid, preds).tolist()

        row = {
            "threshold": round(float(threshold), 2),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
        }

        if best is None or row["accuracy"] > best["accuracy"]:
            best = row

    return best


def main():
    X, y, groups = load_public_json("data/relevant_priors_public.json")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, valid_idx = next(splitter.split(X, y, groups))

    X_train = X.iloc[train_idx]
    y_train = y[train_idx]
    X_valid = X.iloc[valid_idx]
    y_valid = y[valid_idx]

    results = []

    c_values = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    class_weights = [None, "balanced"]

    for C in c_values:
        for class_weight in class_weights:
            model = RelevantPriorsModel(threshold=0.46, C=C)

            # Override classifier so we can test class_weight.
            model.classifier = LogisticRegression(
                max_iter=1000,
                C=C,
                solver="liblinear",
                class_weight=class_weight,
            )

            model.fit(X_train, y_train)
            best = evaluate_model(model, X_valid, y_valid)

            result = {
                "C": C,
                "class_weight": class_weight,
                **best,
            }

            results.append(result)

            print(
                f"C={C}, class_weight={class_weight}, "
                f"threshold={best['threshold']}, "
                f"accuracy={best['accuracy']:.4f}, "
                f"precision={best['precision']:.4f}, "
                f"recall={best['recall']:.4f}, "
                f"f1={best['f1']:.4f}, "
                f"cm={best['confusion_matrix']}"
            )

    results = sorted(results, key=lambda x: x["accuracy"], reverse=True)

    print("\nTop 5:")
    for row in results[:5]:
        print(row)


if __name__ == "__main__":
    main()
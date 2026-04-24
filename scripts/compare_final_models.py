import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

from app.modeling import RelevantPriorsModel


def load_public_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    truth = {
        (str(item["case_id"]), str(item["study_id"])): int(item["is_relevant_to_current"])
        for item in data["truth"]
    }

    rows, labels, groups = [], [], []

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


def evaluate_setting(X, y, groups, C, threshold, seeds):
    rows = []

    for seed in seeds:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, valid_idx = next(splitter.split(X, y, groups))

        model = RelevantPriorsModel(threshold=threshold, C=C)
        model.fit(X.iloc[train_idx], y[train_idx])

        preds = model.predict(X.iloc[valid_idx])
        y_valid = y[valid_idx]

        rows.append(
            {
                "seed": seed,
                "accuracy": accuracy_score(y_valid, preds),
                "precision": precision_score(y_valid, preds, zero_division=0),
                "recall": recall_score(y_valid, preds, zero_division=0),
                "f1": f1_score(y_valid, preds, zero_division=0),
            }
        )

    return pd.DataFrame(rows)


def summarize(name, df):
    print(f"\n{name}")
    print(df.round(4).to_string(index=False))
    print("mean accuracy:", round(df["accuracy"].mean(), 4))
    print("std accuracy: ", round(df["accuracy"].std(), 4))
    print("mean precision:", round(df["precision"].mean(), 4))
    print("mean recall:   ", round(df["recall"].mean(), 4))
    print("mean f1:       ", round(df["f1"].mean(), 4))


def main():
    X, y, groups = load_public_json("data/relevant_priors_public.json")
    seeds = [0, 1, 2, 3, 4]

    aggressive = evaluate_setting(X, y, groups, C=2.0, threshold=0.44, seeds=seeds)
    conservative = evaluate_setting(X, y, groups, C=1.0, threshold=0.54, seeds=seeds)

    summarize("Aggressive: C=2.0 threshold=0.44", aggressive)
    summarize("Conservative: C=1.0 threshold=0.54", conservative)


if __name__ == "__main__":
    main()
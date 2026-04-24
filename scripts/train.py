import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/relevant_priors_public.json")
    parser.add_argument("--out", default="models/relevant_priors_model.joblib")
    parser.add_argument("--threshold", type=float, default=0.44)
    args = parser.parse_args()

    X, y, groups = load_public_json(args.data)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, valid_idx = next(splitter.split(X, y, groups))

    model = RelevantPriorsModel(threshold=args.threshold, C=2.0)
    model.fit(X.iloc[train_idx], y[train_idx])
    valid_pred = model.predict(X.iloc[valid_idx])

    print("Validation accuracy:", round(accuracy_score(y[valid_idx], valid_pred), 4))
    print("Confusion matrix:", confusion_matrix(y[valid_idx], valid_pred).tolist())

    final_model = RelevantPriorsModel(threshold=args.threshold, C=2.0)
    final_model.fit(X, y)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, out)
    print(f"Saved final model to {out}")


if __name__ == "__main__":
    main()

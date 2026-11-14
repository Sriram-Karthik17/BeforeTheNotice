"""Compare held-out logistic scores with the prior-WARN baseline."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from prepare_data import PROCESSED

RESULTS = Path("results")


def main():
    scores = pd.read_csv(PROCESSED / "heldout_scores.csv", dtype={"company_key": str})
    groups = [("overall", "all", scores)]
    groups.extend(("workforce_group", name, group) for name, group in scores.groupby("workforce_group"))
    rows = []
    for dimension, name, group in groups:
        positives = int(group.target.sum())
        for model in ("baseline", "logistic"):
            probability = group[f"{model}_score"]
            both_classes = group.target.nunique() == 2
            rows.append({
                "dimension": dimension, "group": name, "model": model,
                "rows": len(group), "positives": positives, "prevalence": group.target.mean(),
                "average_precision": average_precision_score(group.target, probability) if both_classes else None,
                "roc_auc": roc_auc_score(group.target, probability) if both_classes else None,
            })
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "metrics.csv", index=False)
    scores.to_csv(RESULTS / "predictions.csv", index=False)


if __name__ == "__main__":
    main()

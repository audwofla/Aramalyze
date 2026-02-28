from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from .features import load_team_csv, build_vocab, featurize_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to aram team dataset csv")
    ap.add_argument("--out", default="models/aram_lr.joblib", help="Output model artifact path")
    ap.add_argument("--test_size", type=float, default=0.2, help="Fraction of matches held out")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_team_csv(args.csv)

    match_ids = df["match_id"].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(match_ids)

    split_idx = int((1 - args.test_size) * len(match_ids))
    train_matches = set(match_ids[:split_idx])
    test_matches = set(match_ids[split_idx:])

    train_df = df[df["match_id"].isin(train_matches)].reset_index(drop=True)
    test_df = df[df["match_id"].isin(test_matches)].reset_index(drop=True)

    vocab = build_vocab(train_df)

    X_train, y_train = featurize_df(train_df, vocab)
    X_test, y_test = featurize_df(test_df, vocab)

    model = LogisticRegression(
        max_iter=3000,
        solver="saga",
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("matches_total:", len(match_ids))
    print("matches_train:", len(train_matches))
    print("matches_test:", len(test_matches))
    print("rows_train:", len(train_df))
    print("rows_test:", len(test_df))
    print("features:", X_train.shape[1])
    print("accuracy:", round(accuracy_score(y_test, pred), 4))
    print("roc_auc:", round(roc_auc_score(y_test, proba), 4))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "vocab": vocab,
        "meta": {
            "csv": str(args.csv),
            "matches_total": int(len(match_ids)),
            "matches_train": int(len(train_matches)),
            "matches_test": int(len(test_matches)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "features": int(X_train.shape[1]),
            "test_size": float(args.test_size),
            "seed": int(args.seed),
        },
    }

    joblib.dump(artifact, out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
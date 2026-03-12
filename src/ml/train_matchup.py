from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from .features import Vocab, build_vocab, featurize_df, featurize_team, load_team_csv


def build_matchup_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for match_id, match_df in df.groupby("match_id", sort=False):
        teams = match_df.to_dict("records")
        if len(teams) != 2:
            continue

        team_a, team_b = sorted(teams, key=lambda row: row.get("team_id", 0))

        rows.append(
            {
                "match_id": match_id,
                "team_1_champs": team_a["champs"],
                "team_1_tag_counts": team_a["tag_counts"],
                "team_2_champs": team_b["champs"],
                "team_2_tag_counts": team_b["tag_counts"],
                "team_1_win": int(team_a["win"]),
            }
        )
        rows.append(
            {
                "match_id": match_id,
                "team_1_champs": team_b["champs"],
                "team_1_tag_counts": team_b["tag_counts"],
                "team_2_champs": team_a["champs"],
                "team_2_tag_counts": team_a["tag_counts"],
                "team_1_win": int(team_b["win"]),
            }
        )

    return pd.DataFrame(rows)


def featurize_matchup_df(df: pd.DataFrame, vocab: Vocab) -> tuple[csr_matrix, np.ndarray]:
    if df.empty:
        feature_count = (len(vocab.champ2idx) + len(vocab.tag2idx) + len(vocab.pair2idx)) * 2
        return csr_matrix((0, feature_count), dtype=np.float32), np.array([], dtype=np.int64)

    team_1_df = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "win": df["team_1_win"],
            "champs": df["team_1_champs"],
            "tag_counts": df["team_1_tag_counts"],
        }
    )
    team_2_df = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "win": 1 - df["team_1_win"],
            "champs": df["team_2_champs"],
            "tag_counts": df["team_2_tag_counts"],
        }
    )

    X_team_1, y = featurize_df(team_1_df, vocab)
    X_team_2, _ = featurize_df(team_2_df, vocab)
    X = hstack([X_team_1, X_team_2], format="csr", dtype=np.float32)
    y = df["team_1_win"].to_numpy(dtype=np.int64)
    return X, y


def featurize_matchup(team_1_champs, team_1_tag_counts, team_2_champs, team_2_tag_counts, vocab: Vocab) -> csr_matrix:
    team_1 = featurize_team(team_1_champs, team_1_tag_counts, vocab)
    team_2 = featurize_team(team_2_champs, team_2_tag_counts, vocab)
    return hstack([team_1, team_2], format="csr", dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to aram team dataset csv")
    ap.add_argument("--out", default="models/aram_matchup_lr.joblib", help="Output model artifact path")
    ap.add_argument("--test_size", type=float, default=0.2, help="Fraction of matches held out")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    team_df = load_team_csv(args.csv)

    match_ids = team_df["match_id"].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(match_ids)

    split_idx = int((1 - args.test_size) * len(match_ids))
    train_matches = set(match_ids[:split_idx])
    test_matches = set(match_ids[split_idx:])

    train_team_df = team_df[team_df["match_id"].isin(train_matches)].reset_index(drop=True)
    test_team_df = team_df[team_df["match_id"].isin(test_matches)].reset_index(drop=True)

    vocab = build_vocab(train_team_df)

    train_df = build_matchup_df(train_team_df)
    test_df = build_matchup_df(test_team_df)

    X_train, y_train = featurize_matchup_df(train_df, vocab)
    X_test, y_test = featurize_matchup_df(test_df, vocab)

    model = LogisticRegression(
        C=0.16,
        l1_ratio=0.5,
        max_iter=5000,
        solver="saga",
        verbose=1,
        tol=1e-3,
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
            "mode": "matchup",
        },
    }

    joblib.dump(artifact, out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    main()

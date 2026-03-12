from __future__ import annotations

import argparse
import ast
from typing import Dict, List

import joblib

from .train_matchup import featurize_matchup


def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if s.startswith("["):
        v = ast.literal_eval(s)
        if not isinstance(v, list):
            raise ValueError("champs must be a list")
        return [int(x) for x in v]
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_tag_counts(s: str) -> Dict[str, int]:
    v = ast.literal_eval(s)
    if not isinstance(v, dict):
        raise ValueError("tag_counts must be a dict string")
    return {str(k): int(val) for k, val in v.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/aram_matchup_lr.joblib", help="Path to saved matchup joblib artifact")
    ap.add_argument("--team_1_champs", required=True, help='5 champs, e.g. "[57,63,233,245,555]" or "57,63,233,245,555"')
    ap.add_argument("--team_1_tag_counts", required=True, help='dict string, e.g. "{\'Mage\':2, \'Tank\':1}"')
    ap.add_argument("--team_2_champs", required=True, help='5 champs, e.g. "[12,22,53,79,111]" or "12,22,53,79,111"')
    ap.add_argument("--team_2_tag_counts", required=True, help='dict string, e.g. "{\'Support\':2, \'Tank\':2}"')
    args = ap.parse_args()

    artifact = joblib.load(args.model)
    model = artifact["model"]
    vocab = artifact["vocab"]

    team_1_champs = parse_int_list(args.team_1_champs)
    team_2_champs = parse_int_list(args.team_2_champs)
    if len(team_1_champs) != 5:
        raise ValueError(f"Expected exactly 5 champs for team_1, got {len(team_1_champs)}: {team_1_champs}")
    if len(team_2_champs) != 5:
        raise ValueError(f"Expected exactly 5 champs for team_2, got {len(team_2_champs)}: {team_2_champs}")

    team_1_tag_counts = parse_tag_counts(args.team_1_tag_counts)
    team_2_tag_counts = parse_tag_counts(args.team_2_tag_counts)

    X = featurize_matchup(team_1_champs, team_1_tag_counts, team_2_champs, team_2_tag_counts, vocab)
    win_prob = float(model.predict_proba(X)[0, 1])

    print("team_1_win_prob:", round(win_prob, 4))
    print("team_2_win_prob:", round(1.0 - win_prob, 4))


if __name__ == "__main__":
    main()

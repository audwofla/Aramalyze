from __future__ import annotations

import argparse
import ast
from typing import Dict, List

import joblib

from .features import featurize_team


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
    ap.add_argument("--model", default="models/aram_lr.joblib", help="Path to saved joblib artifact")
    ap.add_argument("--champs", required=True, help='5 champs, e.g. "[57,63,233,245,555]" or "57,63,233,245,555"')
    ap.add_argument("--tag_counts", required=True, help='dict string, e.g. "{\'Mage\':2, \'Tank\':1}"')
    args = ap.parse_args()

    artifact = joblib.load(args.model)
    model = artifact["model"]
    vocab = artifact["vocab"]

    champs = parse_int_list(args.champs)
    if len(champs) != 5:
        raise ValueError(f"Expected exactly 5 champs, got {len(champs)}: {champs}")

    tag_counts = parse_tag_counts(args.tag_counts)

    X = featurize_team(champs, tag_counts, vocab)
    win_prob = float(model.predict_proba(X)[0, 1])

    print("win_prob:", round(win_prob, 4))


if __name__ == "__main__":
    main()
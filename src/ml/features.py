from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack


@dataclass(frozen=True)
class Vocab:
    champ2idx: Dict[int, int]
    tag2idx: Dict[str, int]
    pair2idx: Dict[Tuple[int, int], int]


def _parse_champs(x) -> List[int]:

    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, str):
        v = ast.literal_eval(x)
        if not isinstance(v, list):
            raise ValueError(f"Expected list for champs, got: {type(v)}")
        return [int(i) for i in v]
    raise ValueError(f"Unsupported champs type: {type(x)}")


def _parse_tag_counts(x) -> Dict[str, int]:

    if isinstance(x, dict):
        return {str(k): int(v) for k, v in x.items()}
    if isinstance(x, str):
        v = ast.literal_eval(x)
        if not isinstance(v, dict):
            raise ValueError(f"Expected dict for tag_counts, got: {type(v)}")
        return {str(k): int(val) for k, val in v.items()}
    raise ValueError(f"Unsupported tag_counts type: {type(x)}")


def load_team_csv(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    required = {"match_id", "win", "champs", "tag_counts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    df = df.copy()
    df["champs"] = df["champs"].apply(_parse_champs)
    df["tag_counts"] = df["tag_counts"].apply(_parse_tag_counts)

    # win might be bool, or "True"/"False" strings
    if df["win"].dtype == object:
        df["win"] = df["win"].map({"True": 1, "False": 0, True: 1, False: 0})
    df["win"] = df["win"].astype(int)

    return df


def build_vocab(df: pd.DataFrame) -> Vocab:

    all_champs = sorted({c for row in df["champs"] for c in row})
    champ2idx = {c: i for i, c in enumerate(all_champs)}

    all_tags = sorted({t for row in df["tag_counts"] for t in row.keys()})
    tag2idx = {t: i for i, t in enumerate(all_tags)}

    all_pairs = set()
    for champs in df["champs"]:
        champs_sorted = sorted(champs)
        for a, b in combinations(champs_sorted, 2):
            all_pairs.add((a, b))
    pair2idx = {p: i for i, p in enumerate(sorted(all_pairs))}

    return Vocab(champ2idx=champ2idx, tag2idx=tag2idx, pair2idx=pair2idx)


def featurize_df(df: pd.DataFrame, vocab: Vocab):

    n = len(df)
    n_ch = len(vocab.champ2idx)
    n_tag = len(vocab.tag2idx)
    n_pair = len(vocab.pair2idx)

    rows_ch, cols_ch = [], []
    rows_tag, cols_tag, data_tag = [], [], []
    rows_pair, cols_pair = [], []

    y = df["win"].to_numpy(dtype=np.int64)

    for i, row in enumerate(df.itertuples(index=False)):
        champs = list(row.champs)
        tag_counts = dict(row.tag_counts)

        for c in champs:
            j = vocab.champ2idx.get(int(c))
            if j is not None:
                rows_ch.append(i)
                cols_ch.append(j)

        for tag, count in tag_counts.items():
            k = vocab.tag2idx.get(str(tag))
            if k is not None and count:
                rows_tag.append(i)
                cols_tag.append(k)
                data_tag.append(float(count))

        champs_sorted = sorted(int(c) for c in champs)
        for a, b in combinations(champs_sorted, 2):
            p_idx = vocab.pair2idx.get((a, b))
            if p_idx is not None:
                rows_pair.append(i)
                cols_pair.append(p_idx)

    X_ch = csr_matrix(
        (np.ones(len(rows_ch), dtype=np.float32), (rows_ch, cols_ch)),
        shape=(n, n_ch),
        dtype=np.float32,
    )

    X_tag = csr_matrix(
        (np.array(data_tag, dtype=np.float32), (rows_tag, cols_tag)),
        shape=(n, n_tag),
        dtype=np.float32,
    )

    X_pair = csr_matrix(
        (np.ones(len(rows_pair), dtype=np.float32), (rows_pair, cols_pair)),
        shape=(n, n_pair),
        dtype=np.float32,
    )

    X = hstack([X_ch, X_tag, X_pair], format="csr", dtype=np.float32)
    return X, y


def featurize_team(champs: List[int], tag_counts: Dict[str, int], vocab: Vocab) -> csr_matrix:

    champs_sorted = sorted(int(c) for c in champs)

    n_ch = len(vocab.champ2idx)
    n_tag = len(vocab.tag2idx)
    n_pair = len(vocab.pair2idx)

    cols_ch = []
    for c in champs_sorted:
        j = vocab.champ2idx.get(c)
        if j is not None:
            cols_ch.append(j)
    X_ch = csr_matrix(
        (np.ones(len(cols_ch), dtype=np.float32), ([0] * len(cols_ch), cols_ch)),
        shape=(1, n_ch),
        dtype=np.float32,
    )

    cols_tag = []
    data_tag = []
    for tag, count in tag_counts.items():
        k = vocab.tag2idx.get(str(tag))
        if k is not None and count:
            cols_tag.append(k)
            data_tag.append(float(count))
    X_tag = csr_matrix(
        (np.array(data_tag, dtype=np.float32), ([0] * len(cols_tag), cols_tag)),
        shape=(1, n_tag),
        dtype=np.float32,
    )

    cols_pair = []
    for a, b in combinations(champs_sorted, 2):
        p_idx = vocab.pair2idx.get((a, b))
        if p_idx is not None:
            cols_pair.append(p_idx)
    X_pair = csr_matrix(
        (np.ones(len(cols_pair), dtype=np.float32), ([0] * len(cols_pair), cols_pair)),
        shape=(1, n_pair),
        dtype=np.float32,
    )

    return hstack([X_ch, X_tag, X_pair], format="csr", dtype=np.float32)
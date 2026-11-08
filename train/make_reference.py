"""Taqsimot siljishini kuzatish uchun etalon: o‘quv to‘plamidagi asosiy belgilarning log1p kvantil taqsimoti (models/reference.json)."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sentinel.features import clean  # noqa: E402

FEATS = ["sttl", "dttl", "sbytes", "dbytes", "smean", "dmean", "rate", "sload", "dload", "dur", "ct_srv_src", "ct_dst_sport_ltm"]

if __name__ == "__main__":
    tr = clean(pd.read_csv(os.path.join(ROOT, "data", "unsw", "UNSW_NB15_training-set.csv")))
    ref = {"features": {}}
    for f in FEATS:
        z = np.log1p(np.clip(tr[f].values.astype(float), 0, None))
        edges = np.unique(np.quantile(z, np.linspace(0, 1, 11)))
        cnt = np.histogram(z, np.r_[-np.inf, edges[1:-1], np.inf])[0]
        ref["features"][f] = {"edges": edges.tolist(), "p": (cnt / cnt.sum()).tolist()}
    ref["sttl254"] = round(float((tr.sttl == 254).mean()), 4)
    with open(os.path.join(ROOT, "models", "reference.json"), "w", encoding="utf-8") as fh:
        json.dump(ref, fh)
    print("models/reference.json yozildi")

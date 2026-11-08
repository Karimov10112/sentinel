"""Mahalliy trafikka moslashtirish va oylik qayta o‘qitish (diplom, 3.6-band, C-ilova «C» usuli).

Korxonaning o‘z normal oqimlari (3 baravar vazn bilan) va tahlilchilar yorliqlari o‘quv to‘plamiga qo‘shilib,
binar XGBoost qayta o‘qitiladi; so‘ng chegara mahalliy normal trafikning persentili bo‘yicha kalibrlanadi.

    python train/adapt.py --local samples/pilot_normal.csv
    python train/adapt.py --local local_normal.csv --labels tahlilchi_yorliqlari.csv --percentile 99
Eski model models/backup/ papkasiga saqlanadi.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sentinel.features import Prep, clean  # noqa: E402

M = os.path.join(ROOT, "models")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True, help="mahalliy normal oqimlar (CSV)")
    ap.add_argument("--labels", help="tahlilchi yorliqlari (label ustuni bilan CSV)")
    ap.add_argument("--percentile", type=float, default=95)
    ap.add_argument("--weight", type=float, default=3.0)
    ap.add_argument("--estimators", type=int, default=400)
    ap.add_argument("--variant", choices=["full", "nottl"], default="full")
    a = ap.parse_args()
    tr = clean(pd.read_csv(os.path.join(ROOT, "data", "unsw", "UNSW_NB15_training-set.csv")))
    loc = clean(pd.read_csv(a.local))
    if "label" in loc.columns:
        loc = loc[pd.to_numeric(loc["label"], errors="coerce").fillna(0) == 0]
    parts, ys, ws = [tr], [tr.label.values], [np.ones(len(tr))]
    parts.append(loc)
    ys.append(np.zeros(len(loc), int))
    ws.append(np.full(len(loc), a.weight))
    if a.labels:
        lab = clean(pd.read_csv(a.labels))
        parts.append(lab)
        ys.append(pd.to_numeric(lab["label"], errors="coerce").fillna(0).astype(int).values)
        ws.append(np.full(len(lab), a.weight))
    pfile = "prep.json" if a.variant == "full" else "prep_nottl.json"
    mfile = "xgb_bin.json" if a.variant == "full" else "xgb_bin_nottl.json"
    prep = Prep.load(os.path.join(M, pfile))
    X = np.vstack([prep.transform(p) for p in parts])
    y, w = np.concatenate(ys), np.concatenate(ws)
    print(f"O‘qitish: {len(y)} oqim (mahalliy normal: {len(loc)}, yorliqlar: {len(parts[2]) if a.labels else 0})")
    t = time.time()
    model = xgb.XGBClassifier(n_estimators=a.estimators, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8,
                              n_jobs=2, tree_method="hist", random_state=42).fit(X, y, sample_weight=w)
    print(f"   {time.time() - t:.0f} s")
    thr = float(np.percentile(model.predict_proba(prep.transform(loc))[:, 1], a.percentile))
    bdir = os.path.join(M, "backup", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(bdir, exist_ok=True)
    shutil.copy(os.path.join(M, mfile), bdir)
    shutil.copy(os.path.join(M, "config.json"), bdir)
    model.get_booster().save_model(os.path.join(M, mfile))
    with open(os.path.join(M, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    key = "thr_cal95" if a.variant == "full" else "thr_cal95_nottl"
    cfg[key] = thr
    cfg["adapted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(M, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    print(f"Yangi model saqlandi ({mfile}), kalibrlangan chegara ({a.percentile:.0f}-persentil) = {thr:.4f}; eski model: {bdir}")

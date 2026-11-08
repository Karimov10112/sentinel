"""3-qatlam: SI aniqlash dvigateli (diplom, 3.1-band).

  XGBoost (binar)      — oqimning hujum ekanligi ehtimoli p
  XGBoost (10 sinf)    — hujum turi (taxminiy tur)
  Autoenkoder          — tiklash xatosi e(x), anomaliya ⇔ e(x) > θ_AE  (2.6)
  DNN                  — zaxira va solishtiruvchi model
  TreeSHAP             — qaror sabablari (XGBoost ichki pred_contribs, aniq SHAP qiymatlari)
Modellar portativ formatlarda saqlanadi (XGBoost JSON, NumPy npz, JSON) — pickle ishlatilmaydi.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from . import nn
from .features import CLASSES, UZ, Prep, clean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("SENTINEL_MODELS", os.path.join(ROOT, "models"))


def _booster(path: str) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(path)
    b.set_param({"nthread": 2})
    return b


class Detector:
    def __init__(self, models_dir: str = MODELS_DIR):
        d = models_dir
        with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.prep = Prep.load(os.path.join(d, "prep.json"))
        self.prep_nottl = Prep.load(os.path.join(d, "prep_nottl.json"))
        self.xgb = {"full": _booster(os.path.join(d, "xgb_bin.json")), "nottl": _booster(os.path.join(d, "xgb_bin_nottl.json"))}
        self.xgm = _booster(os.path.join(d, "xgb_multi.json"))
        dim = self.prep.dim
        self.ae = nn.autoencoder(dim).load(dict(np.load(os.path.join(d, "ae.npz"))))
        self.dnn = nn.dnn(dim, 2).load(dict(np.load(os.path.join(d, "dnn.npz"))))
        self.ae_thr = float(self.cfg["ae_thr"])

    def preps(self, variant: str) -> Prep:
        return self.prep_nottl if variant == "nottl" else self.prep

    def score(self, df: pd.DataFrame, variant: str = "full", with_dnn: bool = False) -> dict:
        """Oqimlar → p, hujum turi ehtimolliklari, autoenkoder xatosi (paket bo‘yicha)."""
        df = clean(df)
        X = self.prep.transform(df)
        Xv = X if variant == "full" else self.prep_nottl.transform(df)
        p = self.xgb[variant].predict(xgb.DMatrix(Xv))
        pm = self.xgm.predict(xgb.DMatrix(X)).reshape(len(df), len(CLASSES))
        err = nn.recon_error(self.ae, X)
        out = {"p": p, "pm": pm, "cat": np.array(CLASSES)[1 + pm[:, 1:].argmax(1)], "cls": np.array(CLASSES)[pm.argmax(1)],
               "ae": err, "ae_flag": err > self.ae_thr, "X": Xv}
        if with_dnn:
            out["p_dnn"] = nn.softmax(self.dnn.predict(X))[:, 1]
        return out

    def reasons(self, x_row: np.ndarray, raw: dict, variant: str = "full", k: int = 3) -> list[dict]:
        """SHAP bo‘yicha qarorning k ta asosiy sababi (B-ilova, reasons())."""
        names = self.preps(variant).names
        sv = self.xgb[variant].predict(xgb.DMatrix(x_row[None, :]), pred_contribs=True)[0][:-1]
        top = np.argsort(-sv)[:k]
        out = []
        for i in top:
            f = names[i]
            base = f.split("_")[0] if f.startswith(("proto_", "service_", "state_")) else f
            val = raw.get(f, raw.get(base, "")) if base in ("proto", "service", "state") or f in raw else ""
            if f.startswith(("proto_", "service_", "state_")):
                val = f.split("_", 1)[1]
            out.append({"feature": f, "name": UZ.get(base, f), "value": _fmt(val), "shap": round(float(sv[i]), 3)})
        return out

    def shap_all(self, x_row: np.ndarray, variant: str = "full") -> tuple[list[str], np.ndarray, float]:
        c = self.xgb[variant].predict(xgb.DMatrix(x_row[None, :]), pred_contribs=True)[0]
        return self.preps(variant).names, c[:-1], float(c[-1])


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return int(v) if v.is_integer() else round(v, 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v

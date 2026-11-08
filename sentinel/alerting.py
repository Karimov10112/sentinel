"""4-qatlam: ogohlantirish menejeri (diplom, 3.6-band va B-ilova).

Quvur: oqimlar 500 tadan paketlarda → SI dvigateli → chegara (kalibrlangan) → prioritet (3.2) →
bir xil (hujum turi, xizmat, protokol, manzil) li ogohlantirishlarni W soniyalik oynada insidentga guruhlash →
yangi insident uchun SHAP bo‘yicha 3 ta asosiy sabab, Telegram matni va SIEM JSON yozuvi → yetkazish kanallari.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

from . import priority as PR
from .channels import Channels, siem_record, telegram_text
from .features import clean
from .models import Detector
from .storage import Storage

BATCH = 500


def thresholds(det: Detector, settings: dict) -> dict:
    v = settings.get("variant", "full")
    c = det.cfg
    sfx = "_nottl" if v == "nottl" else ""
    out = {"default": 0.5, "cost": c["thr_cost"], "cal95": c[f"thr_cal95{sfx}"], "cal99": c[f"thr_cal99{sfx}"]}
    loc = settings.get(f"thr_local{sfx}") or ""
    if loc:
        out["local"] = float(loc)
    return out


THR_UZ = {"default": "Asl chegara (0,5)", "cost": "Xatolar narxi bo‘yicha (10:1, validatsiya)",
          "cal95": "Kalibrlangan, 95-persentil (pilot haftasi)", "cal99": "Kalibrlangan, 99-persentil (pilot haftasi)",
          "local": "Mahalliy kalibrlash (korxona trafigi)"}


def active_threshold(det: Detector, settings: dict) -> tuple[float, str]:
    th = thresholds(det, settings)
    mode = settings.get("thr_mode", "cal95")
    if mode not in th:
        mode = "cal95"
    return float(th[mode]), mode


def psi(ref: dict, values: np.ndarray) -> float:
    """Population Stability Index: taqsimot siljishi (> 0,25 — jiddiy siljish, modelni qayta moslash kerak)."""
    edges = np.array(ref["edges"])
    cnt = np.histogram(np.log1p(np.clip(values, 0, None)), np.r_[-np.inf, edges[1:-1], np.inf])[0]
    q = np.maximum(cnt / max(cnt.sum(), 1), 1e-4)
    p = np.maximum(np.array(ref["p"]), 1e-4)
    return float(((q - p) * np.log(q / p)).sum())


class AlertService:
    def __init__(self, store: Storage, det: Detector, channels: Channels, reference: dict | None = None):
        self.store, self.det, self.ch = store, det, channels
        self.ref = reference or {}

    def process(self, raw: pd.DataFrame, source: str = "upload", name: str = "", rate: float = 200.0,
                ts0: float | None = None, send: bool = True) -> dict:
        t_all = time.perf_counter()
        st = self.store.settings()
        variant = st.get("variant", "full")
        thr, mode = active_threshold(self.det, st)
        W = float(st.get("window") or 60)
        cfg = self.ch.cfg()
        df = clean(raw).reset_index(drop=True)
        n = len(df)
        bid = self.store.new_batch(source, name)
        # vaqt: jurnalda bo‘lsa (ts / stime) — o‘sha, aks holda «rate» oqim/s tezlikda keladi deb faraz qilinadi (3.6-band)
        tcol = next((c for c in ("ts", "stime", "timestamp") if c in df.columns), None)
        if tcol:
            ts = pd.to_numeric(df[tcol], errors="coerce").fillna(time.time()).values.astype(float)
        else:
            ts = (ts0 if ts0 is not None else time.time()) + np.arange(n) / rate
        truth = None
        if "attack_cat" in df.columns:
            truth = df["attack_cat"].astype(str).str.strip().replace({"Backdoors": "Backdoor", "nan": "Normal", "": "Normal"}).values
        elif "label" in df.columns:
            truth = np.where(pd.to_numeric(df["label"], errors="coerce").fillna(0).values > 0, "Attack", "Normal")
        srcip = df["srcip"].astype(str).values if "srcip" in df.columns else None
        dstip = df["dstip"].astype(str).values if "dstip" in df.columns else None

        open_: dict[str, tuple[int, float]] = {}
        for r in self.store.recent_incidents(float(ts.min()) - W):
            open_[r["key"]] = (r["id"], r["last_ts"])
        upd: dict[int, dict] = {}
        rows, new_inc, levels_inc = [], [], Counter()
        all_p, all_alert, lv_all = np.zeros(n), np.zeros(n, bool), np.full(n, "", object)
        t_model = t_alert = t_shap = 0.0
        for b0 in range(0, n, BATCH):
            chunk = df.iloc[b0:b0 + BATCH]
            t1 = time.perf_counter()
            sc = self.det.score(chunk, variant)
            t2 = time.perf_counter()
            t_model += t2 - t1
            p, cat, aef, err = sc["p"], sc["cat"], sc["ae_flag"], sc["ae"]
            all_p[b0:b0 + len(chunk)] = p
            idx = np.where((p >= thr) | aef)[0]
            for j in idx:
                i = b0 + j
                c = str(cat[j]) if p[j] >= thr else "Anomaliya"
                s = PR.score(float(p[j]), c, bool(aef[j]), thr)
                lv = PR.level(s)
                if not lv:
                    continue
                all_alert[i] = True
                lv_all[i] = lv
                svc, proto = str(chunk.service.values[j]), str(chunk.proto.values[j])
                dip = dstip[i] if dstip is not None else ""
                key = f"{c}|{svc}|{proto}|{dip}"
                o = open_.get(key)
                is_true = None if truth is None else truth[i] != "Normal"
                if o and ts[i] - o[1] <= W:
                    iid = o[0]
                    u = upd.setdefault(iid, {"last": ts[i], "n": 0, "s": 0.0, "lv": "P4", "p": 0.0, "t": 0, "f": 0, "ips": set()})
                    u["last"] = max(u["last"], ts[i])
                    u["n"] += 1
                    if s > u["s"]:
                        u["s"], u["lv"] = s, lv
                    u["p"] = max(u["p"], float(p[j]))
                    if is_true is not None:
                        u["t" if is_true else "f"] += 1
                    if srcip is not None:
                        u["ips"].add(srcip[i])
                    open_[key] = (iid, ts[i])
                else:
                    ts_ = time.perf_counter()
                    rawrow = {k: (v.item() if hasattr(v, "item") else v) for k, v in chunk.iloc[j].to_dict().items()}
                    reasons = self.det.reasons(sc["X"][j], rawrow, variant) if c != "Anomaliya" else \
                        self._ae_reasons(chunk.iloc[[j]], rawrow)
                    inc = {"key": key, "first_ts": float(ts[i]), "last_ts": float(ts[i]), "created": time.time(), "category": c,
                           "service": svc, "proto": proto, "dstip": dip, "srcips": [srcip[i]] if srcip is not None else [],
                           "count": 1, "max_score": s, "level": lv, "max_p": round(float(p[j]), 4), "reasons": reasons,
                           "sample": {k: v for k, v in rawrow.items() if k not in ("label", "attack_cat")},
                           "truth_true": int(bool(is_true)), "truth_false": int(is_true is False), "batch_id": bid,
                           "variant": variant, "thr": thr}
                    iid = self.store.create_incident(inc)
                    inc["id"], inc["window"] = iid, int(W)
                    text, rec = telegram_text(inc), siem_record(inc)
                    self.store.set_incident_text(iid, text, rec)
                    self.ch.dispatch(inc, text, rec, cfg, telegram=send)
                    t_shap += time.perf_counter() - ts_
                    open_[key] = (iid, ts[i])
                    new_inc.append(iid)
                    levels_inc[lv] += 1
                rows.append((float(ts[i]), bid, open_[key][0], round(float(p[j]), 4), round(float(err[j]), 5), int(aef[j]), s, lv, c,
                             proto, svc, str(chunk.state.values[j]), srcip[i] if srcip is not None else None, dip or None,
                             float(chunk.sbytes.values[j]), float(chunk.dbytes.values[j]), float(chunk.sttl.values[j]),
                             None if truth is None else truth[i]))
            t_alert += time.perf_counter() - t2
        t_db = time.perf_counter()
        self.store.add_alerts(rows)
        for iid, u in upd.items():
            self.store.update_incident_stats(iid, u["last"], u["n"], u["s"], u["lv"], u["p"], u["t"], u["f"], sorted(u["ips"]))
        for iid in set(new_inc) | set(upd):                                     # yakuniy son bilan matnni yangilash
            inc = self.store.incident(iid)
            inc["window"] = int(W)
            self.store.set_incident_text(iid, telegram_text(inc), siem_record(inc))
        t_db = time.perf_counter() - t_db
        total = time.perf_counter() - t_all

        lv_cnt = Counter(lv_all[all_alert])
        summary = {"flows": n, "alerts": int(all_alert.sum()), "levels": dict(lv_cnt), "incidents_new": len(new_inc),
                   "incidents_levels": dict(levels_inc), "incidents_touched": len(set(new_inc) | set(upd)),
                   "thr": round(thr, 4), "thr_mode": mode, "variant": variant, "window": W, "telegram": send,
                   "ms_model_1000": round(t_model / max(n, 1) * 1e6, 1),
                   "ms_alert_1000": round((t_alert - t_shap) / max(n, 1) * 1e6, 1),
                   "ms_incident": round(t_shap / max(len(new_inc), 1) * 1000, 2), "ms_db": round(t_db * 1000, 1),
                   "throughput": int(n / max(total - t_shap - t_db, 1e-6)), "seconds": round(total, 2)}
        model_al = all_p >= thr
        summary["alerts_model"] = int(model_al.sum())
        summary["alerts_ae_only"] = int((all_alert & ~model_al).sum())
        if truth is not None:
            att = truth != "Normal"
            al = model_al                      # 3.8-jadvaldagidek: kalibrlangan chegaradan o‘tgan ogohlantirishlar
            summary["truth"] = {
                "attacks": int(att.sum()), "true_alerts": int((al & att).sum()),
                "precision": round(float(att[al].mean()), 4) if al.any() else 0.0,
                "recall": round(float(al[att].mean()), 4) if att.any() else 0.0,
                "fpr": round(float(al[np.logical_not(att)].mean()), 4) if np.logical_not(att).any() else 0.0,
                "by_level": {k: round(float(att[lv_all == k].mean()), 4) for k in ("P1", "P2", "P3", "P4") if (lv_all == k).any()},
                "by_category": {c: round(float(al[truth == c].mean()), 4) for c in sorted(set(truth)) if c != "Normal"},
                "ae_only_true": int((all_alert & ~model_al & att).sum()),
            }
        summary["drift"] = self.drift(df, all_p, thr)
        self.store.finish_batch(bid, summary)
        return {"batch_id": bid, **summary}

    def _ae_reasons(self, row_df: pd.DataFrame, raw: dict) -> list[dict]:
        """Faqat autoenkoder anomaliyasi uchun: eng katta tiklash xatosiga ega 3 ta belgi."""
        from .features import UZ
        X = self.det.prep.transform(row_df)
        r = self.det.ae.predict(X)
        e = ((r - X) ** 2)[0]
        names = self.det.prep.names
        out = []
        for i in np.argsort(-e)[:3]:
            f = names[i]
            base = f.split("_")[0] if f.startswith(("proto_", "service_", "state_")) else f
            out.append({"feature": f, "name": UZ.get(base, f) + " (AE xatosi)", "value": raw.get(f, f.split("_", 1)[-1]),
                        "shap": round(float(e[i]), 3)})
        return out

    def drift(self, df: pd.DataFrame, p: np.ndarray, thr: float) -> dict:
        out = {"sttl254": round(float((df.sttl.values == 254).mean()), 4), "alert_rate_10k": round(float((p >= thr).mean() * 1e4), 1)}
        feats = self.ref.get("features", {})
        if feats and len(df) >= 200:
            ps = {f: round(psi(r, df[f].values), 4) for f, r in feats.items() if f in df.columns}
            out["psi"] = ps
            out["psi_max"] = max(ps.values()) if ps else 0.0
        return out


def load_reference(models_dir: str) -> dict:
    p = os.path.join(models_dir, "reference.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _nan(v):
    return None if isinstance(v, float) and math.isnan(v) else v

"""UNSW-NB15 da SI modellarini o‘qitish va diplomning 3-bobidagi barcha tajribalarni takrorlash (A- va C-ilovalar).

Ishlatish:   python train/train_models.py            (2 yadroli kompyuterda ≈ 15–25 daqiqa)
             python train/train_models.py --fast     (RF, LR, IF, ko‘p sinfli DNN va namuna hajmi tajribasisiz)

Natijalar:
  models/prep.json, prep_nottl.json     — belgilarni tayyorlash parametrlari
  models/xgb_bin.json, xgb_bin_nottl.json — binar XGBoost (barcha belgilar / TTL belgilarisiz)
  models/xgb_multi.json                 — 10 sinfli XGBoost (hujum turi)
  models/dnn.npz, models/ae.npz         — DNN (zaxira model) va autoenkoder (anomaliya signali)
  models/config.json                    — chegaralar (asl, xatolar narxi, kalibrlangan 95/99-persentil, autoenkoder)
  models/metrics.json                   — 3.3–3.8-jadvallar va rasmlar uchun natijalar
  samples/pilot_normal.csv              — «pilot haftasi» mahalliy normal trafik namunasi (kalibrlash uchun)
  samples/stream_20000.csv              — prototip sinovi uchun oqimlar (kalibrlashga kirmagan)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sentinel import nn  # noqa: E402
from sentinel.features import CLASSES, NUM, SEV, TTL, Prep, clean  # noqa: E402

D = os.path.join(ROOT, "data", "unsw")
OUT = os.path.join(ROOT, "models")
SAMPLES = os.path.join(ROOT, "samples")
SEED = 42
LEVELS = [(0.85, "P1"), (0.60, "P2"), (0.40, "P3"), (0.25, "P4")]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def metrics(y, pred, score=None, train_s=None, ms=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    out = {"accuracy": round(float((tp + tn) / len(y)), 4), "precision": round(float(P), 4), "recall": round(float(R), 4),
           "f1": round(float(2 * P * R / max(P + R, 1e-12)), 4), "fpr": round(float(fp / max(fp + tn, 1)), 4),
           "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
    if score is not None:
        out["roc_auc"] = round(float(roc_auc_score(y, score)), 4)
        f, t, _ = roc_curve(y, score)
        idx = np.unique(np.r_[np.linspace(0, len(f) - 1, 70).astype(int)])
        out["roc"] = {"fpr": [round(float(v), 4) for v in f[idx]], "tpr": [round(float(v), 4) for v in t[idx]]}
    if train_s is not None:
        out["train_s"] = round(train_s, 1)
    if ms is not None:
        out["ms_per_1000"] = round(ms, 2)
    return out


def timed(fn, X):
    t = time.perf_counter()
    r = fn(X)
    return r, (time.perf_counter() - t) / len(X) * 1e6


def make_xgb(**kw):
    p = dict(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8, n_jobs=2,
             tree_method="hist", random_state=SEED, eval_metric="logloss")
    p.update(kw)
    return xgb.XGBClassifier(**p)


def shap_values(model, X):
    """TreeSHAP (XGBoost ichki pred_contribs) — shap kutubxonasisiz aniq SHAP qiymatlari."""
    c = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return c[:, :-1]


def priority(p, cat, ae_flag, thr=0.5):
    w = np.array([SEV.get(c, 0.3) for c in cat])
    s = np.where(p >= thr, p * (0.5 + 0.5 * w), 0.0) + 0.25 * ae_flag * (p < thr)
    lv = np.full(len(p), "", object)
    for t, code in reversed(LEVELS):
        lv[s >= t] = code
    return s, lv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SAMPLES, exist_ok=True)
    R: dict = {}

    log("1) UNSW-NB15 yuklanmoqda")
    tr = clean(pd.read_csv(os.path.join(D, "UNSW_NB15_training-set.csv")))
    te = clean(pd.read_csv(os.path.join(D, "UNSW_NB15_testing-set.csv")))
    for d in (tr, te):
        d["attack_cat"] = d["attack_cat"].astype(str).str.strip().replace({"Backdoors": "Backdoor"})
    trn, val = train_test_split(tr, test_size=0.1, stratify=tr.attack_cat, random_state=SEED)
    prep = Prep().fit(trn)
    Xtr, Xva, Xte = prep.transform(trn), prep.transform(val), prep.transform(te)
    ytr, yva, yte = trn.label.values, val.label.values, te.label.values
    cmap = {c: i for i, c in enumerate(CLASSES)}
    mtr, mva, mte = trn.attack_cat.map(cmap).values, val.attack_cat.map(cmap).values, te.attack_cat.map(cmap).values
    FEAT = prep.names
    R["dataset"] = {"n_train": len(trn), "n_val": len(val), "n_test": len(te), "n_feat": int(Xtr.shape[1]),
                    "train_attack": round(float(tr.label.mean()), 4), "test_attack": round(float(te.label.mean()), 4),
                    "train_cnt": tr.attack_cat.value_counts().to_dict(), "test_cnt": te.attack_cat.value_counts().to_dict()}
    log("   belgilar:", Xtr.shape[1], "| o‘quv", len(trn), "validatsiya", len(val), "sinov", len(te))
    prep.save(os.path.join(OUT, "prep.json"))

    # ------------------------------------------------------------------ 2) binar tasnif (3.3-jadval)
    B, P = {}, {}
    if not args.fast:
        log("2) Logistic Regression")
        t = time.perf_counter()
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, ytr)
        tt = time.perf_counter() - t
        P["Logistic Regression"], ms = timed(lambda X: lr.predict_proba(X)[:, 1], Xte)
        B["Logistic Regression"] = metrics(yte, (P["Logistic Regression"] >= 0.5).astype(int), P["Logistic Regression"], tt, ms)
        log("   Random Forest")
        t = time.perf_counter()
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=2, random_state=SEED).fit(Xtr, ytr)
        tt = time.perf_counter() - t
        P["Random Forest"], ms = timed(lambda X: rf.predict_proba(X)[:, 1], Xte)
        B["Random Forest"] = metrics(yte, (P["Random Forest"] >= 0.5).astype(int), P["Random Forest"], tt, ms)
        del rf
    log("   XGBoost")
    t = time.perf_counter()
    xg = make_xgb().fit(Xtr, ytr)
    tt = time.perf_counter() - t
    P["XGBoost"], ms = timed(lambda X: xg.predict_proba(X)[:, 1], Xte)
    B["XGBoost"] = metrics(yte, (P["XGBoost"] >= 0.5).astype(int), P["XGBoost"], tt, ms)
    xg.get_booster().save_model(os.path.join(OUT, "xgb_bin.json"))

    log("   DNN (256–128–64)")
    t = time.perf_counter()
    dnn, dhist = nn.fit_classifier(nn.dnn(Xtr.shape[1], 2), Xtr, ytr, Xva, yva, 2, log=log)
    tt = time.perf_counter() - t
    P["DNN (MLP)"], ms = timed(lambda X: nn.softmax(dnn.predict(X))[:, 1], Xte)
    B["DNN (MLP)"] = metrics(yte, (P["DNN (MLP)"] >= 0.5).astype(int), P["DNN (MLP)"], tt, ms)
    np.savez_compressed(os.path.join(OUT, "dnn.npz"), **dnn.state())

    log("   Autoenkoder (faqat normal trafik)")
    t = time.perf_counter()
    ae, ahist = nn.fit_autoencoder(nn.autoencoder(Xtr.shape[1]), Xtr[ytr == 0], Xva[yva == 0], log=log)
    tt = time.perf_counter() - t
    ae_thr = float(np.percentile(nn.recon_error(ae, Xva[yva == 0]), 95))
    E, ms = timed(lambda X: nn.recon_error(ae, X), Xte)
    B["Autoenkoder"] = metrics(yte, (E > ae_thr).astype(int), E, tt, ms)
    np.savez_compressed(os.path.join(OUT, "ae.npz"), **ae.state())
    if not args.fast:
        log("   Isolation Forest")
        t = time.perf_counter()
        iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=2).fit(Xtr[ytr == 0])
        tt = time.perf_counter() - t
        if_thr = float(np.percentile(-iso.score_samples(Xtr[ytr == 0]), 95))
        S, ms = timed(lambda X: -iso.score_samples(X), Xte)
        B["Isolation Forest"] = metrics(yte, (S > if_thr).astype(int), S, tt, ms)
    hyb = ((P["XGBoost"] >= 0.5) | (E > ae_thr)).astype(int)
    B["Gibrid (XGBoost + AE)"] = metrics(yte, hyb)
    for k, v in B.items():
        log(f"   {k:22s} Acc={v['accuracy']:.3f} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} FPR={v['fpr']:.3f}"
            f" AUC={v.get('roc_auc', 0):.3f}")
    R["binary"] = B
    R["curves"] = {"dnn": [[round(a, 4), round(b, 4)] for a, b in dhist], "ae": [[round(a, 5), round(b, 5)] for a, b in ahist]}
    bins = np.logspace(np.log10(max(E.min(), 1e-4)), np.log10(E.max()), 40)
    R["ae_hist"] = {"bins": [round(float(b), 5) for b in bins], "normal": np.histogram(E[yte == 0], bins)[0].tolist(),
                    "attack": np.histogram(E[yte == 1], bins)[0].tolist(), "thr": round(ae_thr, 5)}
    cat_rate = {}
    for name, pred in (("XGBoost", P["XGBoost"] >= 0.5), ("DNN (MLP)", P["DNN (MLP)"] >= 0.5), ("Autoenkoder", E > ae_thr),
                       ("Gibrid (XGBoost + AE)", hyb.astype(bool))):
        cat_rate[name] = {c: round(float(pred[te.attack_cat.values == c].mean()), 4) for c in CLASSES[1:]}
    if "Random Forest" in P:
        cat_rate["Random Forest"] = {c: round(float((P["Random Forest"] >= 0.5)[te.attack_cat.values == c].mean()), 4)
                                     for c in CLASSES[1:]}
    R["by_category"] = cat_rate

    # ------------------------------------------------------------------ 3) ko‘p sinfli (3.4-jadval)
    log("3) Ko‘p sinfli XGBoost (sinf vaznlari √(N/(K·n_k)))")
    cnt = np.bincount(mtr, minlength=len(CLASSES))
    cw = np.sqrt(len(mtr) / (len(CLASSES) * np.maximum(cnt, 1)))
    xgm = make_xgb(objective="multi:softprob", eval_metric="mlogloss").fit(Xtr, mtr, sample_weight=cw[mtr])
    xgm.get_booster().save_model(os.path.join(OUT, "xgb_multi.json"))
    pm = xgm.predict_proba(Xte)
    pred_m = pm.argmax(1)

    def mc(pred):
        pr, rc, f1, _ = precision_recall_fscore_support(mte, pred, labels=range(len(CLASSES)), zero_division=0)
        cm = confusion_matrix(mte, pred, labels=range(len(CLASSES)))
        return {"per_class": {c: {"n": int((mte == i).sum()), "precision": round(float(pr[i]), 3), "recall": round(float(rc[i]), 3),
                                  "f1": round(float(f1[i]), 3)} for i, c in enumerate(CLASSES)},
                "accuracy": round(float(accuracy_score(mte, pred)), 4),
                "macro_f1": round(float(f1_score(mte, pred, average="macro")), 4),
                "weighted_f1": round(float(f1_score(mte, pred, average="weighted")), 4),
                "confusion_pct": (cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100).round(1).tolist()}
    R["multiclass"] = {"XGBoost": mc(pred_m)}
    att = mte != 0
    R["multiclass"]["XGBoost"]["type_correct_in_attacks"] = round(float((pred_m[att] == mte[att]).mean()), 4)
    if not args.fast:
        log("   Ko‘p sinfli DNN")
        dm, _ = nn.fit_classifier(nn.dnn(Xtr.shape[1], len(CLASSES)), Xtr, mtr, Xva, mva, len(CLASSES), wts=cw, log=log)
        R["multiclass"]["DNN (MLP)"] = mc(nn.softmax(dm.predict(Xte)).argmax(1))
        del dm
    log(f"   XGBoost: acc={R['multiclass']['XGBoost']['accuracy']} macroF1={R['multiclass']['XGBoost']['macro_f1']}")

    # ------------------------------------------------------------------ 4) SHAP va taqsimot siljishi (3.5-band)
    log("4) SHAP (TreeSHAP, 3 000 oqim) va taqsimot siljishi")
    rs = np.random.RandomState(SEED)
    sidx = rs.choice(len(te), 3000, replace=False)
    sv = shap_values(xg, Xte[sidx])
    mabs = np.abs(sv).mean(0)
    order = np.argsort(-mabs)[:15]
    R["shap"] = {"global": [{"feature": FEAT[i], "mean_abs": round(float(mabs[i]), 3)} for i in order],
                 "beeswarm": [{"feature": FEAT[i], "shap": [round(float(v), 3) for v in sv[:400, i]],
                               "value": [round(float(v), 3) for v in Xte[sidx[:400], i]]} for i in order[:10]]}
    nt_tr, nt_te = (tr.label == 0), (te.label == 0)
    fp_mask = (P["XGBoost"] >= 0.5) & (yte == 0)
    R["shift"] = {"sttl254_train_normal": round(float((tr.sttl[nt_tr] == 254).mean()), 4),
                  "sttl254_test_normal": round(float((te.sttl[nt_te] == 254).mean()), 4),
                  "fp_with_sttl254": round(float((te.sttl.values[fp_mask] == 254).mean()), 4), "fp": int(fp_mask.sum())}

    # kalibrlash bo‘linmasi (C-ilova): sinovdagi normal oqimlarning 20% i — «pilot haftasi»
    rs7 = np.random.RandomState(7)
    norm_idx = np.where(yte == 0)[0]
    cal = rs7.choice(norm_idx, int(0.2 * len(norm_idx)), replace=False)
    ev = np.setdiff1d(np.arange(len(te)), cal)
    te.iloc[cal].drop(columns=["id"], errors="ignore").to_csv(os.path.join(SAMPLES, "pilot_normal.csv"), index=False)

    log("   TTL belgilarisiz modellar (3.5-jadval)")
    abl = {}
    variants = [("Barcha belgilar", NUM), ("sttl siz", [c for c in NUM if c != "sttl"]),
                ("TTL belgilarisiz (sttl, dttl, ct_state_ttl)", [c for c in NUM if c not in TTL])]
    for name, num in variants:
        if num is NUM:
            pv, model, pp, Xe = P["XGBoost"], xg, prep, Xte
        else:
            pp = Prep(num).fit(trn)
            model = make_xgb().fit(pp.transform(trn), ytr)
            Xe = pp.transform(te)
            pv = model.predict_proba(Xe)[:, 1]
        thr = float(np.percentile(pv[cal], 95))
        a, c = metrics(yte[ev], (pv[ev] >= 0.5).astype(int)), metrics(yte[ev], (pv[ev] >= thr).astype(int))
        top = pp.names[int(np.abs(shap_values(model, Xe[sidx[:1000]])).mean(0).argmax())]
        abl[name] = {"roc_auc": round(float(roc_auc_score(yte[ev], pv[ev])), 4), "fpr_05": a["fpr"], "f1_05": a["f1"],
                     "fpr_cal": c["fpr"], "recall_cal": c["recall"], "thr_cal": round(thr, 4), "top_feature": top}
        if name.startswith("TTL"):
            pp.save(os.path.join(OUT, "prep_nottl.json"))
            model.get_booster().save_model(os.path.join(OUT, "xgb_bin_nottl.json"))
            p_nottl = pv
    R["ttl_ablation"] = abl

    # ------------------------------------------------------------------ 5) chegara tanlash (3.1) va moslashtirish (3.6-jadval)
    log("5) Chegara: xatolar narxi C(θ) = 10·FN + FP")
    pva = xg.predict_proba(Xva)[:, 1]
    grid = np.round(np.linspace(0.01, 0.99, 99), 2)
    costs = [10 * ((pva < t) & (yva == 1)).sum() + ((pva >= t) & (yva == 0)).sum() for t in grid]
    th_cost = float(grid[int(np.argmin(costs))])
    mc_ = metrics(yte, (P["XGBoost"] >= th_cost).astype(int))
    R["cost_threshold"] = {"theta": th_cost, "test_recall": mc_["recall"], "test_fpr": mc_["fpr"],
                           "val_cost": [int(c) for c in costs], "grid": grid.tolist()}
    curve = []
    for t in np.round(np.linspace(0.05, 0.995, 40), 3):
        m_ = metrics(yte, (P["XGBoost"] >= t).astype(int))
        curve.append({"thr": float(t), "recall": m_["recall"], "precision": m_["precision"], "fpr": m_["fpr"],
                      "per10k": round(float((P["XGBoost"] >= t).mean() * 10000))})
    R["threshold_curve"] = curve

    log("   Mahalliy trafikka moslashtirish (C-ilova)")
    p = P["XGBoost"]
    ad = {"n_cal": int(len(cal)), "n_eval": int(len(ev)), "n_eval_normal": int((yte[ev] == 0).sum())}
    ad["A. Asl model"] = dict(metrics(yte[ev], (p[ev] >= 0.5).astype(int)), thr=0.5)
    thr95, thr99 = float(np.percentile(p[cal], 95)), float(np.percentile(p[cal], 99))
    ad["B. Kalibrlangan chegara (95-persentil)"] = dict(metrics(yte[ev], (p[ev] >= thr95).astype(int)), thr=thr95)
    ad["B. Kalibrlangan chegara (99-persentil)"] = dict(metrics(yte[ev], (p[ev] >= thr99).astype(int)), thr=thr99)

    def retrain(n):
        sub = cal[:n]
        Xa = np.vstack([Xtr, Xte[sub]])
        ya = np.r_[ytr, np.zeros(len(sub), int)]
        w = np.r_[np.ones(len(ytr)), np.full(len(sub), 3.0)]
        return make_xgb().fit(Xa, ya, sample_weight=w).predict_proba(Xte)[:, 1]
    p2 = retrain(len(cal))
    ad["C. Mahalliy trafik bilan qayta o‘qitish"] = dict(metrics(yte[ev], (p2[ev] >= 0.5).astype(int)), thr=0.5)
    t2 = float(np.percentile(p2[cal], 99))
    ad["D. Qayta o‘qitish + kalibrlangan chegara (99)"] = dict(metrics(yte[ev], (p2[ev] >= t2).astype(int)), thr=t2)
    R["adapt"] = ad
    sizes = []
    if not args.fast:
        for n in (370, 1850, 3700):
            pn = retrain(n)
            sizes.append({"n": n, "fpr": metrics(yte[ev], (pn[ev] >= 0.5).astype(int))["fpr"]})
    sizes.append({"n": int(len(cal)), "fpr": ad["C. Mahalliy trafik bilan qayta o‘qitish"]["fpr"]})
    R["adapt_sizes"] = [{"n": 0, "fpr": ad["A. Asl model"]["fpr"]}] + sizes
    for k, v in ad.items():
        if isinstance(v, dict):
            log(f"   {k:46s} thr={v['thr']:.3f} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} FPR={v['fpr']:.3f} FP={v['fp']}")

    # ------------------------------------------------------------------ 6) prioritet (3.2) va guruhlash
    log("6) Prioritetlash va guruhlash")
    att_cat = np.array(CLASSES)[1 + pm[:, 1:].argmax(1)]        # hujum bo‘lsa — eng ehtimolli hujum turi
    ae_flag = (E > ae_thr).astype(float)
    S, LV = priority(p, att_cat, ae_flag, 0.5)
    pr_tab = {}
    for code, ch in (("P1", "Telegram darhol + SOAR"), ("P2", "Telegram darhol"), ("P3", "e-pochta dayjesti"), ("P4", "faqat SIEM")):
        mk = LV == code
        pr_tab[code] = {"alerts": int(mk.sum()), "true": int((yte[mk] == 1).sum()), "false": int((yte[mk] == 0).sum()),
                        "precision": round(float((yte[mk] == 1).mean()) if mk.any() else 0, 4), "channel": ch}
    allk = LV != ""
    R["priority"] = {"levels": pr_tab, "attacks_in_p1p2": round(float(np.isin(LV[yte == 1], ["P1", "P2"]).mean()), 4),
                     "fp_in_p3p4": round(float(np.isin(LV[(yte == 0) & allk], ["P3", "P4"]).mean()), 4),
                     "fp_share_all": round(float((yte[allk] == 0).mean()), 4),
                     "fp_share_p1p2": round(float((yte[np.isin(LV, ['P1', 'P2'])] == 0).mean()), 4)}
    rsg = np.random.RandomState(SEED)
    order_ = rsg.permutation(len(te))
    ts = np.arange(len(te)) / 200.0                               # 200 oqim/s deb faraz qilinadi
    grp = []
    a_idx = order_[allk[order_]]
    a_ts = ts[np.where(allk[order_])[0]]
    for W in (10, 30, 60, 120, 300):
        last, n_inc = {}, 0
        for i, t in zip(a_idx, a_ts):
            k = (att_cat[i] if p[i] >= 0.5 else "Anomaliya", te.service.values[i], te.proto.values[i])
            if k not in last or t - last[k] > W:
                n_inc += 1
            last[k] = t
        grp.append({"window": W, "incidents": n_inc})
    R["grouping"] = {"alerts": int(allk.sum()), "by_window": grp}

    # ------------------------------------------------------------------ 7) prototip sinovi (3.8-jadval)
    log("7) Prototip: kalibrlashga kirmagan 20 000 oqim")
    rsp = np.random.RandomState(11)
    pidx = rsp.choice(ev, 20000, replace=False)
    st = te.iloc[pidx].drop(columns=["id"], errors="ignore").reset_index(drop=True)
    st.to_csv(os.path.join(SAMPLES, "stream_20000.csv"), index=False)
    t1 = time.perf_counter()
    Xs = prep.transform(st)
    t2_ = time.perf_counter()
    ps = xg.predict_proba(Xs)[:, 1]
    pms = xgm.predict_proba(Xs)
    es = nn.recon_error(ae, Xs)
    t3 = time.perf_counter()
    cats = np.array(CLASSES)[1 + pms[:, 1:].argmax(1)]
    Ss, Ls = priority(ps, cats, (es > ae_thr).astype(float), thr95)
    al = ps >= thr95
    ys = st.label.values
    last, incs = {}, []
    tss = np.arange(len(st)) / 200.0
    for i in np.where(al)[0]:
        k = (cats[i], st.service.values[i], st.proto.values[i])
        if k not in last or tss[i] - last[k][0] > 60:
            incs.append(i)
            last[k] = (tss[i], len(incs) - 1)
        else:
            last[k] = (tss[i], last[k][1])
    t4 = time.perf_counter()
    t5 = time.perf_counter()
    for i in incs:
        shap_values(xg, Xs[i:i + 1])
    t6 = time.perf_counter()
    inc_lv = Ls[incs]
    R["prototype"] = {"flows": len(st), "attacks": int(ys.sum()), "alerts": int(al.sum()), "true_alerts": int((ys[al] == 1).sum()),
                      "precision": round(float((ys[al] == 1).mean()), 4), "recall": round(float(al[ys == 1].mean()), 4),
                      "incidents": len(incs), "inc_p1": int((inc_lv == "P1").sum()), "inc_p2": int((inc_lv == "P2").sum()),
                      "prec_p1": round(float((ys[al & (Ls == 'P1')] == 1).mean()), 4),
                      "prec_p2": round(float((ys[al & (Ls == 'P2')] == 1).mean()), 4),
                      "ms_features": round((t2_ - t1) / len(st) * 1000 * 1000, 1), "ms_model": round((t3 - t2_) / len(st) * 1e6, 1),
                      "ms_alert": round((t4 - t3) / len(st) * 1e6, 1), "ms_shap_incident": round((t6 - t5) / max(len(incs), 1) * 1000, 2),
                      "throughput": int(len(st) / max(t4 - t1, 1e-6)), "thr": round(thr95, 4)}
    log("  ", R["prototype"])

    cfg = {"thr_default": 0.5, "thr_cost": th_cost, "thr_cal95": round(thr95, 6), "thr_cal99": round(thr99, 6),
           "ae_thr": ae_thr, "thr_cal95_nottl": abl["TTL belgilarisiz (sttl, dttl, ct_state_ttl)"]["thr_cal"],
           "thr_cal99_nottl": round(float(np.percentile(p_nottl[cal], 99)), 6), "n_feat": int(Xtr.shape[1]),
           "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "xgboost": xgb.__version__}
    with open(os.path.join(OUT, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    R["config"] = cfg
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, default=float)
    log("Tayyor:", OUT)


if __name__ == "__main__":
    main()

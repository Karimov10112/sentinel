"""SentinelAI avtomatik testlari:  python -m pytest -q"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sentinel import nn  # noqa: E402
from sentinel import priority as PR  # noqa: E402
from sentinel.alerting import AlertService, active_threshold, load_reference, psi  # noqa: E402
from sentinel.channels import Channels, siem_record, telegram_text  # noqa: E402
from sentinel.features import CLASSES, NUM, Prep, clean  # noqa: E402
from sentinel.models import MODELS_DIR, Detector  # noqa: E402
from sentinel.storage import Storage  # noqa: E402

SAMPLES = os.path.join(ROOT, "samples")


@pytest.fixture(scope="session")
def det():
    return Detector()


@pytest.fixture(scope="session")
def stream():
    return pd.read_csv(os.path.join(SAMPLES, "stream_20000.csv"))


@pytest.fixture()
def svc(tmp_path, det):
    st = Storage(str(tmp_path / "t.db"))
    return AlertService(st, det, Channels(st, str(tmp_path / "ai-alerts.json")), load_reference(MODELS_DIR))


# ---------------------------------------------------------------- formulalar
def test_priority_formula():
    assert PR.score(1.0, "Backdoor", False, 0.5) == 1.0
    assert PR.score(0.9, "Reconnaissance", False, 0.5) == pytest.approx(0.9 * 0.65)
    assert PR.score(0.3, "Exploits", True, 0.5) == 0.25            # faqat autoenkoder → P4
    assert PR.score(0.3, "Exploits", False, 0.5) == 0.0
    assert [PR.level(s) for s in (0.9, 0.85, 0.7, 0.5, 0.3, 0.1)] == ["P1", "P1", "P2", "P3", "P4", ""]


def test_prep_matches_formula():
    df = pd.DataFrame({**{c: [0.0, 10.0, 100.0] for c in NUM}, "proto": ["tcp", "udp", "tcp"],
                       "service": ["-", "dns", "http"], "state": ["FIN", "INT", "CON"]})
    p = Prep().fit(df)
    X = p.transform(df)
    z = np.log1p([0.0, 10.0, 100.0])
    assert X[:, 0] == pytest.approx((z - z.mean()) / z.std(), abs=1e-5)
    assert X.shape[1] == p.dim == len(NUM) + 2 + 3 + 3
    unk = p.transform(pd.DataFrame({**{c: [1.0] for c in NUM}, "proto": ["xyz"], "service": ["-"], "state": ["FIN"]}))
    assert unk[0, len(NUM):len(NUM) + 2].sum() == 0                  # handle_unknown='ignore'
    assert Prep.from_json(json.loads(json.dumps(p.to_json()))).transform(df) == pytest.approx(X)


def test_numpy_network_gradients():
    rng = np.random.default_rng(0)
    net = nn.Net([nn.Linear(4, 6, rng), nn.BatchNorm(6), nn.ReLU(), nn.Linear(6, 3, rng)])
    for layer in net.layers:
        for p in layer.params:
            setattr(layer, p, getattr(layer, p).astype(np.float64))
    x, y = rng.normal(size=(8, 4)), np.array([0, 1, 2, 1, 0, 2, 1, 0])

    def loss():
        pr = nn.softmax(net.forward(x, True))
        return -np.log(pr[np.arange(8), y]).mean()
    pr = nn.softmax(net.forward(x, True))
    g = pr.copy()
    g[np.arange(8), y] -= 1
    net.backward(g / 8)
    L, e = net.layers[0], 1e-6
    an = L.dW[2, 3]
    L.W[2, 3] += e
    a = loss()
    L.W[2, 3] -= 2 * e
    b = loss()
    assert an == pytest.approx((a - b) / (2 * e), rel=1e-4)


def test_psi():
    ref = {"edges": [0, 1, 2, 3, 4], "p": [0.25, 0.25, 0.25, 0.25]}
    same = np.expm1(np.random.default_rng(1).uniform(0, 4, 5000))
    assert psi(ref, same) < 0.02
    assert psi(ref, np.expm1(np.full(1000, 3.5))) > 1


# ---------------------------------------------------------------- modellar
def test_models_reproduce_diploma():
    with open(os.path.join(MODELS_DIR, "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    assert m["dataset"]["n_feat"] == 194
    assert m["binary"]["XGBoost"]["f1"] == pytest.approx(0.893, abs=0.005)
    assert m["binary"]["XGBoost"]["roc_auc"] == pytest.approx(0.983, abs=0.003)
    a = m["adapt"]
    assert a["B. Kalibrlangan chegara (95-persentil)"]["fpr"] < 0.06 < a["A. Asl model"]["fpr"]
    assert m["multiclass"]["XGBoost"]["accuracy"] == pytest.approx(0.732, abs=0.01)
    assert m["prototype"]["precision"] > 0.95


def test_detector_scores(det, stream):
    sub = stream.sample(2000, random_state=3)
    sc = det.score(sub)
    y = sub.label.values
    thr = det.cfg["thr_cal95"]
    pred = sc["p"] >= thr
    assert (y[pred] == 1).mean() > 0.93 and pred[y == 1].mean() > 0.85
    assert sc["pm"].shape == (2000, len(CLASSES)) and np.allclose(sc["pm"].sum(1), 1, atol=1e-4)
    gen = sub.attack_cat.astype(str).str.strip().values == "Generic"
    assert (sc["cat"][gen] == "Generic").mean() > 0.9


def test_nottl_variant_does_not_use_ttl(det, stream):
    assert not {"sttl", "dttl", "ct_state_ttl"} & set(det.prep_nottl.num)
    row = stream.iloc[[0]].copy()
    a = det.score(row, "nottl")["p"][0]
    row["sttl"], row["dttl"] = 1, 1
    assert det.score(row, "nottl")["p"][0] == pytest.approx(a)


def test_shap_reasons(det, stream):
    row = stream[stream.attack_cat.str.strip() == "Exploits"].iloc[[0]]
    sc = det.score(row)
    raw = clean(row).iloc[0].to_dict()
    r = det.reasons(sc["X"][0], raw)
    assert len(r) == 3 and all({"feature", "name", "value", "shap"} <= set(x) for x in r)
    names, sv, base = det.shap_all(sc["X"][0])
    logit = np.log(sc["p"][0] / (1 - sc["p"][0]))
    assert sv.sum() + base == pytest.approx(logit, abs=1e-3)       # SHAP qiymatlari yig‘indisi = model natijasi


# ---------------------------------------------------------------- ogohlantirish xizmati
def test_alert_service_prototype(svc, stream):
    res = svc.process(stream.head(4000), "test", "test", send=False)
    t = res["truth"]
    assert t["precision"] > 0.94 and t["recall"] > 0.85
    assert 0 < res["incidents_new"] < res["alerts"] / 10                # guruhlash
    inc = svc.store.incidents(level="P1")[0]
    assert inc["telegram"].startswith("[P1 – KRITIK]") and len(inc["reasons"]) == 3
    with open(svc.ch.cfg()["siem_path"], encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == res["incidents_new"] and json.loads(lines[0])["ai_alert"]["source"] == "SentinelAI"
    out = svc.store.outbox()
    assert {o["channel"] for o in out} >= {"siem", "telegram", "digest", "soar"}
    assert all(o["status"] == "preview" for o in out if o["channel"] == "telegram")


def test_grouping_window(svc, stream):
    sub = stream.head(3000)
    svc.store.set("window", 10)
    a = svc.process(sub, "t", "w10", send=False)["incidents_new"]
    svc.store.reset()
    svc.store.set("window", 300)
    b = svc.process(sub, "t", "w300", send=False)["incidents_new"]
    assert a > b


def test_threshold_modes(svc, det):
    st = svc.store.settings()
    for mode in ("default", "cost", "cal95", "cal99"):
        st["thr_mode"] = mode
        thr, m = active_threshold(det, st)
        assert m == mode and 0 < thr < 1
    st["thr_mode"] = "local"                                        # kalibrlanmagan → cal95
    assert active_threshold(det, st)[1] == "cal95"


def test_ae_only_alert_is_p4(svc, det):
    df = pd.DataFrame([{c: 1e7 for c in NUM} | {"proto": "zzz", "service": "-", "state": "INT"}])
    sc = det.score(df)
    if sc["p"][0] < active_threshold(det, svc.store.settings())[0]:
        assert sc["ae_flag"][0]
        res = svc.process(df, "t", "ae", send=False)
        assert res["levels"] == {"P4": 1}


def test_ips_and_timestamps(svc):
    df = pd.read_csv(os.path.join(SAMPLES, "demo_with_ips.csv"))
    res = svc.process(df, "t", "ips", send=False)
    inc = svc.store.incidents(limit=500)
    assert res["incidents_new"] > 0 and any(i["dstip"] for i in inc) and any(i["srcips"] for i in inc)
    assert "Manba:" in inc[0]["telegram"]


def test_telegram_text_and_siem():
    inc = {"id": 1, "level": "P1", "category": "Exploits", "max_p": 0.99, "max_score": 0.94, "service": "http", "proto": "tcp",
           "reasons": [{"feature": "sttl", "name": "manba TTL qiymati", "value": 62, "shap": 2.1}], "count": 5, "first_ts": 1e9,
           "srcips": [], "dstip": ""}
    t = telegram_text(inc)
    assert "[P1 – KRITIK] Exploits" in t and "manba TTL qiymati = 62 (+2.1)" in t and "Tavsiya:" in t
    assert siem_record(inc)["priority"] == "P1"


def test_telegram_not_configured(tmp_path):
    st = Storage(str(tmp_path / "c.db"))
    status, err = Channels(st, str(tmp_path / "s.json")).to_telegram("x", {"telegram_token": "", "telegram_chat": ""})
    assert status == "preview" and "sozlanmagan" in err


# ---------------------------------------------------------------- veb-konsol va API
@pytest.fixture()
def client(tmp_path):
    from app import create_app
    app = create_app(str(tmp_path / "web.db"))
    app.config["TESTING"] = True
    c = app.test_client()
    c.get("/")
    with c.session_transaction() as s:
        c.csrf = s["csrf"]
    return c


def test_pages(client):
    for u in ["/", "/incidents", "/analyze", "/stream", "/calibration", "/channels", "/monitoring", "/models", "/api", "/about"]:
        assert client.get(u).status_code == 200, u
    assert client.get("/incidents/999").status_code == 404
    assert client.post("/stream/run", data={"n": 100}).status_code == 400          # CSRF


def test_stream_and_incident_pages(client):
    r = client.post("/stream/run", data={"csrf": client.csrf, "n": "2000", "seed": "3", "rate": "200"})
    assert r.status_code == 302
    page = client.get(r.headers["Location"]).get_data(as_text=True)
    assert "Precision" in page
    for u in ["/", "/incidents", "/incidents/1", "/monitoring", "/channels"]:
        assert client.get(u).status_code == 200, u
    r = client.post("/incidents/1/verdict", data={"csrf": client.csrf, "verdict": "tp", "status": "closed", "analyst": "A"})
    assert r.status_code == 302 and "incident_id" in client.get("/labels.csv").get_data(as_text=True)


def test_analyze_page(client):
    for c in ("Exploits", "Normal"):
        page = client.post("/analyze", data={"csrf": client.csrf, "sample": c}).get_data(as_text=True)
        assert "SHAP" in page and "Haqiqiy yorliq" in page
    bad = client.post("/analyze", data={"csrf": client.csrf, "flow": "{buzuq"}).get_data(as_text=True)
    assert "o‘qib bo‘lmadi" in bad


def test_calibration(client):
    r = client.post("/calibration/run", data={"csrf": client.csrf, "percentile": "95"})
    assert r.status_code == 302
    h = client.get("/api/v1/health").get_json()
    assert h["threshold_mode"] == "local" and 0.8 < h["threshold"] < 1
    client.post("/calibration/settings", data={"csrf": client.csrf, "thr_mode": "default", "variant": "nottl", "window": "60"})
    h = client.get("/api/v1/health").get_json()
    assert h["threshold"] == 0.5 and h["variant"] == "nottl"


def test_api(client, stream):
    rows = stream.head(300).drop(columns=["label", "attack_cat"]).to_dict("records")
    r = client.post("/api/v1/score", data=json.dumps(rows), content_type="application/json")
    items = r.get_json()["items"]
    assert len(items) == 300 and {"p", "category", "priority"} <= set(items[0])
    r = client.post("/api/v1/flows?telegram=0", data=stream.head(500).to_csv(index=False), content_type="text/csv")
    assert r.status_code == 200 and r.get_json()["flows"] == 500
    assert client.get("/api/v1/incidents").get_json()["items"]
    assert client.post("/api/v1/flows", data="buzuq{", content_type="text/plain").status_code == 400


def test_upload_file(client):
    data = open(os.path.join(SAMPLES, "demo_with_ips.csv"), "rb").read()
    r = client.post("/stream/run", data={"csrf": client.csrf, "file": (io.BytesIO(data), "demo.csv")},
                    content_type="multipart/form-data")
    assert r.status_code == 302 and "/batches/" in r.headers["Location"]

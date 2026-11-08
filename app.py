"""SentinelAI — sun’iy intellekt asosida kiberhujumlarni aniqlash va ogohlantirish tizimi: SOC veb-konsoli (Flask).

Ishga tushirish:
    pip install -r requirements.txt
    python app.py                         # http://127.0.0.1:5000

Muhit o‘zgaruvchilari (ixtiyoriy):
    SENTINEL_API_KEY            REST API kaliti (X-API-Key sarlavhasi)
    SOC_BOT_TOKEN, SOC_CHAT_ID  Telegram bot tokeni va SOC guruhi (konsoldagi sozlamalardan ham kiritiladi)
    SENTINEL_DB                 SQLite fayli (standart: instance/sentinel.db)
    SENTINEL_SECRET, PORT
"""
from __future__ import annotations

import io
import json
import os
import secrets
import time
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
from flask import Flask, Response, abort, flash, jsonify, redirect, render_template, request, session, url_for

from sentinel import __version__
from sentinel.alerting import THR_UZ, AlertService, active_threshold, load_reference, thresholds
from sentinel.channels import Channels, siem_record, telegram_text
from sentinel.features import CLASS_UZ, CLASSES, GROUP, MITRE, RECOMMEND, SEV, UZ, clean
from sentinel.models import MODELS_DIR, Detector
from sentinel import priority as PR
from sentinel.storage import Storage

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(BASE, "samples")
STATUS_UZ = {"sent": "Yuborildi", "preview": "Ko‘rib chiqish (kanal sozlanmagan)", "failed": "Xato", "queued": "Navbatda",
             "digested": "Dayjestga kiritildi", "done": "Bajarildi"}
CHANNEL_UZ = {"siem": "SIEM (Wazuh)", "telegram": "Telegram", "digest": "Dayjest navbati", "email": "E-pochta", "soar": "SOAR"}


def read_table(raw: bytes, filename: str = "") -> pd.DataFrame:
    text = raw.decode("utf-8-sig", errors="replace")
    s = text.lstrip()
    if s.startswith("[") or s.startswith("{"):
        if s.startswith("["):
            return pd.DataFrame(json.loads(s))
        if s.startswith('{"flows"') or s.startswith('{ "flows"'):
            return pd.DataFrame(json.loads(s)["flows"])
        return pd.DataFrame([json.loads(line) for line in s.splitlines() if line.strip()])
    return pd.read_csv(io.StringIO(text))


def create_app(db_path: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(SECRET_KEY=os.environ.get("SENTINEL_SECRET") or secrets.token_hex(16), MAX_CONTENT_LENGTH=200 * 1024 * 1024)
    app.json.ensure_ascii = False
    inst = os.path.join(BASE, "instance")
    store = Storage(db_path or os.environ.get("SENTINEL_DB") or os.path.join(inst, "sentinel.db"))
    det = Detector(MODELS_DIR)
    ch = Channels(store, os.path.join(os.path.dirname(os.path.abspath(store.path)), "ai-alerts.json"))
    svc = AlertService(store, det, ch, load_reference(MODELS_DIR))
    app.extensions.update(store=store, detector=det, service=svc, channels=ch)
    api_key = os.environ.get("SENTINEL_API_KEY")
    with open(os.path.join(MODELS_DIR, "metrics.json"), encoding="utf-8") as f:
        M = json.load(f)
    stream_cache: dict = {}

    def stream_df() -> pd.DataFrame:
        if "df" not in stream_cache:
            stream_cache["df"] = pd.read_csv(os.path.join(SAMPLES, "stream_20000.csv"))
        return stream_cache["df"]

    # ------------------------------------------------------------------ yordamchilar
    @app.context_processor
    def inject():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        st = store.settings()
        thr, mode = active_threshold(det, st)
        with store.conn() as c:
            open_p1 = c.execute("SELECT COUNT(*) FROM incidents WHERE status='open' AND level IN ('P1','P2')").fetchone()[0]
        return {"csrf_token": session["csrf"], "version": __version__, "THR": thr, "THR_MODE": mode, "THR_UZ": THR_UZ,
                "VARIANT": st.get("variant", "full"), "LEVEL_NAME": PR.LEVEL_NAME, "CHANNEL": PR.CHANNEL, "CLASS_UZ": CLASS_UZ,
                "RECOMMEND": RECOMMEND, "MITRE": MITRE, "GROUP": GROUP, "SEV": SEV, "UZ": UZ, "open_p12": open_p1,
                "STATUS_UZ": STATUS_UZ, "CHANNEL_UZ": CHANNEL_UZ}

    def check_csrf():
        if request.form.get("csrf") != session.get("csrf"):
            abort(400, "CSRF tekshiruvi muvaffaqiyatsiz — sahifani yangilab, qayta urinib ko‘ring")

    def require_api_key(fn):
        @wraps(fn)
        def w(*a, **kw):
            if api_key and request.headers.get("X-API-Key") != api_key:
                return jsonify({"error": "unauthorized"}), 401
            return fn(*a, **kw)
        return w

    @app.template_filter("t")
    def fmt_t(v):
        return datetime.fromtimestamp(float(v)).strftime("%Y-%m-%d %H:%M:%S") if v else "—"

    @app.template_filter("ts")
    def fmt_ts(v):
        return datetime.fromtimestamp(float(v)).strftime("%d.%m %H:%M:%S") if v else "—"

    @app.template_filter("hm")
    def fmt_hm(v):
        return datetime.fromtimestamp(float(v)).strftime("%H:%M") if v else ""

    @app.template_filter("pct")
    def pct(v, d=1):
        return "—" if v is None else f"{float(v) * 100:.{d}f}%".replace(".", ",")

    @app.template_filter("dec")
    def dec(v, d=3):
        return "—" if v is None else f"{float(v):.{d}f}".replace(".", ",")

    @app.template_filter("pjson")
    def pjson(v):
        return json.dumps(v, ensure_ascii=False, indent=1)

    @app.template_filter("num")
    def num(v):
        return f"{int(v or 0):,}".replace(",", " ")

    # ------------------------------------------------------------------ boshqaruv paneli
    @app.route("/")
    def dashboard():
        s = store.stats()
        incs = [i for i in store.incidents(status="open", limit=60)][:10]
        tl: dict[int, dict] = {}
        for r in s["timeline"]:
            tl.setdefault(r["bucket"], {"P1": 0, "P2": 0, "P3": 0, "P4": 0})[r["level"]] = r["n"]
        timeline = sorted(tl.items())[-60:]
        return render_template("dashboard.html", s=s, incs=incs, timeline=timeline, batches=store.batches(5),
                               ch=ch.cfg(), M=M)

    # ------------------------------------------------------------------ insidentlar
    @app.route("/incidents")
    def incidents():
        lv, st, vd = request.args.get("level", ""), request.args.get("status", ""), request.args.get("verdict", "")
        page = max(int(request.args.get("page", 1) or 1), 1)
        rows = store.incidents(lv, st, vd, limit=101, offset=(page - 1) * 100)
        return render_template("incidents.html", rows=rows[:100], more=len(rows) > 100, page=page, level=lv, status=st, verdict=vd)

    @app.route("/incidents/<int:iid>")
    def incident(iid):
        inc = store.incident(iid)
        if not inc:
            abort(404, "Insident topilmadi")
        shap = None
        if inc["category"] != "Anomaliya" and inc["sample"]:
            sc = det.score(pd.DataFrame([inc["sample"]]), inc.get("variant") or "full")
            names, sv, base = det.shap_all(sc["X"][0], inc.get("variant") or "full")
            order = np.argsort(-np.abs(sv))[:10]
            shap = {"items": [{"f": names[i], "name": UZ.get(names[i].split("_")[0] if names[i].startswith(("proto_", "service_", "state_")) else names[i], names[i]),
                               "v": round(float(sv[i]), 3)} for i in order], "base": round(base, 3),
                    "pm": {c: round(float(x), 4) for c, x in zip(CLASSES, sc["pm"][0])}, "ae": float(sc["ae"][0])}
        return render_template("incident.html", inc=inc, alerts=store.incident_alerts(iid, 300), shap=shap,
                               msgs=store.outbox_for(iid))

    @app.post("/incidents/<int:iid>/verdict")
    def incident_verdict(iid):
        check_csrf()
        v = request.form.get("verdict", "")
        if v not in ("", "tp", "fp"):
            abort(400)
        status = request.form.get("status", "open")
        store.set_verdict(iid, v, request.form.get("analyst", ""), request.form.get("note", ""),
                          status if status in ("open", "closed") else "open")
        flash("Tahlilchi bahosi saqlandi — u oylik qayta o‘qitish uchun yorliq sifatida eksport qilinadi", "ok")
        return redirect(url_for("incident", iid=iid))

    @app.post("/incidents/<int:iid>/resend")
    def incident_resend(iid):
        check_csrf()
        inc = store.incident(iid)
        if not inc:
            abort(404)
        st, err = ch.to_telegram(inc["telegram"], ch.cfg())
        store.add_outbox("telegram", iid, inc["level"], inc["telegram"], st, err)
        flash(f"Telegram: {STATUS_UZ.get(st, st)}{' — ' + err if err else ''}", "ok" if st == "sent" else "error")
        return redirect(url_for("incident", iid=iid))

    @app.route("/labels.csv")
    def labels_csv():
        rows = [dict(i["sample"], attack_cat_pred=i["category"], verdict=i["verdict"], incident_id=i["id"],
                     label=1 if i["verdict"] == "tp" else 0) for i in store.incidents(limit=100000) if i["verdict"]]
        return Response(pd.DataFrame(rows).to_csv(index=False), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=tahlilchi_yorliqlari.csv"})

    # ------------------------------------------------------------------ bitta oqimni tahlil qilish
    @app.route("/analyze", methods=["GET", "POST"])
    def analyze():
        res, row, err = None, None, None
        cats = ["Normal"] + CLASSES[1:]
        if request.method == "POST":
            check_csrf()
            try:
                if request.form.get("sample"):
                    df = stream_df()
                    c = request.form["sample"]
                    pool = df[df.attack_cat.astype(str).str.strip() == c] if c in cats else df
                    row = pool.sample(1, random_state=int(time.time() * 1000) % 100000).iloc[0].to_dict()
                else:
                    t = request.form.get("flow", "").strip()
                    if not t:
                        raise ValueError("Oqim kiritilmagan")
                    if t.startswith("{"):
                        row = json.loads(t)
                    else:
                        lines = [x for x in t.splitlines() if x.strip()]
                        if len(lines) >= 2:
                            row = pd.read_csv(io.StringIO("\n".join(lines[:2]))).iloc[0].to_dict()
                        else:
                            hdr = pd.read_csv(os.path.join(SAMPLES, "stream_20000.csv"), nrows=0).columns
                            row = pd.read_csv(io.StringIO(lines[0]), header=None, names=list(hdr)).iloc[0].to_dict()
                res = analyze_row(row)
            except (ValueError, KeyError, json.JSONDecodeError, pd.errors.ParserError) as e:
                err = f"Oqimni o‘qib bo‘lmadi: {e}"
        return render_template("analyze.html", res=res, row=row, err=err, cats=cats)

    def analyze_row(row: dict) -> dict:
        st = store.settings()
        variant = st.get("variant", "full")
        thr, mode = active_threshold(det, st)
        df = pd.DataFrame([row])
        sc = det.score(df, variant, with_dnn=True)
        p, cat, aef = float(sc["p"][0]), str(sc["cat"][0]), bool(sc["ae_flag"][0])
        c = cat if p >= thr else ("Anomaliya" if aef else "Normal")
        s = PR.score(p, c, aef, thr)
        lv = PR.level(s)
        names, sv, base = det.shap_all(sc["X"][0], variant)
        order = np.argsort(-np.abs(sv))[:12]
        raw = {k: (v.item() if hasattr(v, "item") else v) for k, v in clean(df).iloc[0].to_dict().items()}
        reasons = det.reasons(sc["X"][0], raw, variant)
        inc = {"category": c, "level": lv or "—", "max_p": p, "max_score": s, "service": raw.get("service"), "proto": raw.get("proto"),
               "reasons": reasons, "count": 1, "first_ts": time.time(), "srcips": [str(row["srcip"])] if row.get("srcip") else [],
               "dstip": str(row.get("dstip") or ""), "window": 60}
        return {"p": p, "p_dnn": float(sc["p_dnn"][0]), "cat": cat, "category": c, "ae": float(sc["ae"][0]), "ae_flag": aef,
                "ae_thr": det.ae_thr, "score": s, "level": lv, "thr": thr, "mode": mode, "variant": variant,
                "pm": sorted(((cl, float(x)) for cl, x in zip(CLASSES, sc["pm"][0])), key=lambda t: -t[1]),
                "shap": [{"f": names[i], "v": float(sv[i]), "x": raw.get(names[i], names[i].split("_", 1)[-1] if "_" in names[i] and names[i].split("_")[0] in ("proto", "service", "state") else "")} for i in order],
                "base": base, "truth": str(row.get("attack_cat", "")).strip() or None,
                "telegram": telegram_text(inc) if lv else None, "siem": siem_record(inc) if lv else None}

    # ------------------------------------------------------------------ oqimni qayta ishlash (prototip)
    @app.route("/stream")
    def stream():
        return render_template("stream.html", batches=store.batches(30), M=M)

    @app.post("/stream/run")
    def stream_run():
        check_csrf()
        send = bool(request.form.get("send"))
        rate = float(request.form.get("rate", 200) or 200)
        f = request.files.get("file")
        if f and f.filename:
            try:
                df = read_table(f.read(), f.filename)
            except (ValueError, json.JSONDecodeError, pd.errors.ParserError, UnicodeDecodeError) as e:
                flash(f"Faylni o‘qib bo‘lmadi: {e}", "error")
                return redirect(url_for("stream"))
            if df.empty:
                flash("Fayl bo‘sh", "error")
                return redirect(url_for("stream"))
            name = f"Fayl: {f.filename}"
        else:
            n = min(max(int(request.form.get("n", 5000) or 5000), 100), 20000)
            seed = int(request.form.get("seed", 1) or 1)
            df = stream_df().sample(n, random_state=seed) if n < 20000 else stream_df()
            name = f"UNSW-NB15 sinov oqimi: {n} ta oqim (kalibrlashga kirmagan)"
        res = svc.process(df, source="stream", name=name, rate=rate, send=send)
        flash(f"Qayta ishlandi: {res['flows']} oqim → {res['alerts']} ogohlantirish → {res['incidents_new']} yangi insident", "ok")
        return redirect(url_for("batch", bid=res["batch_id"]))

    @app.route("/batches/<int:bid>")
    def batch(bid):
        b = store.batch(bid)
        if not b:
            abort(404)
        incs = sorted((i for i in store.incidents(limit=100000) if i["batch_id"] == bid), key=lambda i: i["id"])[:30]
        return render_template("batch.html", b=b, incs=incs)

    # ------------------------------------------------------------------ kalibrlash va model sozlamalari
    @app.route("/calibration")
    def calibration():
        st = store.settings()
        return render_template("calibration.html", th=thresholds(det, st), st=st, cals=store.calibrations(), M=M,
                               cfg=det.cfg)

    @app.post("/calibration/settings")
    def calibration_settings():
        check_csrf()
        mode = request.form.get("thr_mode", "cal95")
        variant = request.form.get("variant", "full")
        if variant in ("full", "nottl"):
            store.set("variant", variant)
        if mode in thresholds(det, store.settings()):
            store.set("thr_mode", mode)
        w = int(request.form.get("window", 60) or 60)
        store.set("window", min(max(w, 5), 3600))
        flash("Sozlamalar saqlandi", "ok")
        return redirect(url_for("calibration"))

    @app.post("/calibration/run")
    def calibration_run():
        """C-ilova: mahalliy normal trafik ehtimolliklarining persentili bo‘yicha chegara."""
        check_csrf()
        pctl = float(request.form.get("percentile", 95) or 95)
        f = request.files.get("file")
        if f and f.filename:
            try:
                df = read_table(f.read(), f.filename)
            except (ValueError, json.JSONDecodeError, pd.errors.ParserError) as e:
                flash(f"Faylni o‘qib bo‘lmadi: {e}", "error")
                return redirect(url_for("calibration"))
            source = f"Fayl: {f.filename}"
        else:
            df = pd.read_csv(os.path.join(SAMPLES, "pilot_normal.csv"))
            source = "Pilot haftasi namunasi (UNSW-NB15 sinovdagi normal oqimlarning 20%)"
        if len(df) < 200:
            flash("Kalibrlash uchun kamida 200 ta normal oqim kerak", "error")
            return redirect(url_for("calibration"))
        if "label" in df.columns and (pd.to_numeric(df["label"], errors="coerce") > 0).any():
            df = df[pd.to_numeric(df["label"], errors="coerce").fillna(0) == 0]
            flash("Fayldagi hujum yorliqli qatorlar kalibrlashdan chiqarildi", "ok")
        res = {}
        for v in ("full", "nottl"):
            p = det.score(df, v)["p"]
            res[v] = float(np.percentile(p, pctl))
            store.set("thr_local" if v == "full" else "thr_local_nottl", res[v])
        store.set("thr_mode", "local")
        v = store.get("variant", "full")
        store.add_calibration(source, len(df), pctl, res[v], v)
        flash(f"Chegara kalibrlandi: θ = {res[v]:.4f} ({pctl:.0f}-persentil, {len(df)} oqim). Faol rejim — mahalliy kalibrlash.", "ok")
        return redirect(url_for("calibration"))

    # ------------------------------------------------------------------ kanallar
    @app.route("/channels")
    def channels():
        cfg = ch.cfg()
        return render_template("channels.html", cfg=cfg, st=store.settings(), out=store.outbox(limit=150),
                               queued=len(store.outbox(channel="digest", status="queued", limit=10000)),
                               env_token=bool(os.environ.get("SOC_BOT_TOKEN")))

    @app.post("/channels/save")
    def channels_save():
        check_csrf()
        for k in ("telegram_chat", "siem_path", "smtp_host", "smtp_port", "smtp_user", "smtp_to"):
            store.set(k, request.form.get(k, "").strip())
        for k in ("telegram_token", "smtp_pass"):
            v = request.form.get(k, "").strip()
            if v:
                store.set(k, v)
            if request.form.get(f"clear_{k}"):
                store.set(k, "")
        store.set("soar", "on" if request.form.get("soar") else "off")
        flash("Kanallar sozlamalari saqlandi", "ok")
        return redirect(url_for("channels"))

    @app.post("/channels/test")
    def channels_test():
        check_csrf()
        text = f"SentinelAI: sinov xabari ({time.strftime('%Y-%m-%d %H:%M:%S')}). Kanal ishlayapti."
        st, err = ch.to_telegram(text, ch.cfg())
        store.add_outbox("telegram", None, "TEST", text, st, err)
        flash(f"Telegram: {STATUS_UZ.get(st, st)}{' — ' + err if err else ''}", "ok" if st == "sent" else "error")
        return redirect(url_for("channels"))

    @app.post("/channels/digest")
    def channels_digest():
        check_csrf()
        st, err, n = ch.send_digest()
        flash("Dayjest navbati bo‘sh" if st == "empty" else f"Dayjest ({n} ta insident): {STATUS_UZ.get(st, st)}{' — ' + err if err else ''}",
              "ok" if st in ("sent", "preview", "empty") else "error")
        return redirect(url_for("channels"))

    @app.route("/channels/siem.json")
    def channels_siem():
        p = ch.cfg()["siem_path"]
        if not os.path.exists(p):
            abort(404, "SIEM jurnali hali yaratilmagan")
        with open(p, encoding="utf-8") as fh:
            lines = fh.readlines()[-500:]
        return Response("".join(lines), mimetype="application/x-ndjson; charset=utf-8")

    # ------------------------------------------------------------------ monitoring
    @app.route("/monitoring")
    def monitoring():
        bs = [b for b in reversed(store.batches(200)) if b["summary"].get("flows")]
        s = store.stats()
        v = s["verdicts"]
        fb_prec = v.get("tp", 0) / max(v.get("tp", 0) + v.get("fp", 0), 1) if v else None
        return render_template("monitoring.html", bs=bs, s=s, fb_prec=fb_prec, ref=svc.ref, M=M)

    # ------------------------------------------------------------------ modellar
    @app.route("/models")
    def models():
        return render_template("models.html", M=M, cfg=det.cfg, CLASSES=CLASSES)

    # ------------------------------------------------------------------ boshqa
    @app.post("/reset")
    def reset():
        check_csrf()
        store.reset()
        flash("Ogohlantirishlar, insidentlar, xabarlar va tahlillar tozalandi (sozlamalar saqlandi)", "ok")
        return redirect(url_for("dashboard"))

    @app.route("/samples/<path:name>")
    def sample(name):
        p = os.path.join(SAMPLES, os.path.basename(name))
        if not os.path.isfile(p):
            abort(404)
        with open(p, "rb") as fh:
            return Response(fh.read(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={os.path.basename(p)}"})

    @app.route("/api")
    def api_docs():
        return render_template("api.html", api_key_set=bool(api_key))

    @app.route("/about")
    def about():
        return render_template("about.html", M=M)

    @app.get("/api/v1/health")
    def api_health():
        st = store.settings()
        thr, mode = active_threshold(det, st)
        return jsonify({"status": "ok", "version": __version__, "threshold": thr, "threshold_mode": mode,
                        "variant": st.get("variant"), "models_trained": det.cfg.get("trained_at")})

    @app.post("/api/v1/flows")
    @require_api_key
    def api_flows():
        """Oqimlar: JSON ro‘yxat, {"flows": [...]}, JSON-lines yoki CSV. Real vaqt rejimida Telegram yoqilgan."""
        try:
            df = read_table(request.get_data())
        except (ValueError, json.JSONDecodeError, pd.errors.ParserError) as e:
            return jsonify({"error": "parse", "message": str(e)}), 400
        if df.empty:
            return jsonify({"error": "empty"}), 400
        send = request.args.get("telegram", "1") != "0"
        res = svc.process(df, source="api", name=f"API: {len(df)} oqim", send=send, ts0=time.time())
        return jsonify({k: res[k] for k in ("batch_id", "flows", "alerts", "levels", "incidents_new", "thr", "variant")})

    @app.post("/api/v1/score")
    @require_api_key
    def api_score():
        """Saqlamasdan baholash: har bir oqim uchun p, hujum turi, autoenkoder, prioritet."""
        try:
            df = read_table(request.get_data())
        except (ValueError, json.JSONDecodeError, pd.errors.ParserError) as e:
            return jsonify({"error": "parse", "message": str(e)}), 400
        st = store.settings()
        thr, _ = active_threshold(det, st)
        sc = det.score(df, st.get("variant", "full"))
        out = []
        for i in range(len(df)):
            p, aef = float(sc["p"][i]), bool(sc["ae_flag"][i])
            c = str(sc["cat"][i]) if p >= thr else ("Anomaliya" if aef else "Normal")
            s = PR.score(p, c, aef, thr)
            out.append({"p": round(p, 4), "category": c, "ae_error": round(float(sc["ae"][i]), 5), "ae_anomaly": aef,
                        "score": s, "priority": PR.level(s) or None})
        return jsonify({"threshold": thr, "items": out})

    @app.get("/api/v1/incidents")
    @require_api_key
    def api_incidents():
        return jsonify({"items": store.incidents(request.args.get("level", ""), request.args.get("status", ""),
                                                 limit=min(int(request.args.get("limit", 100)), 1000))})

    @app.get("/api/v1/incidents/<int:iid>")
    @require_api_key
    def api_incident(iid):
        inc = store.incident(iid)
        return (jsonify(inc), 200) if inc else (jsonify({"error": "not_found"}), 404)

    @app.get("/api/v1/stats")
    @require_api_key
    def api_stats():
        return jsonify(store.stats())

    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(413)
    def err(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": e.name, "message": getattr(e, "description", "")}), e.code
        return render_template("error.html", e=e), e.code

    return app


if __name__ == "__main__":
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)

"""Ma’lumotlar ombori (SQLite): ogohlantirishlar, insidentlar, xabarlar navbati, tahlillar, sozlamalar, tahlilchi baholari."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager

_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, batch_id INTEGER, incident_id INTEGER,
  p REAL, ae REAL, ae_flag INTEGER, score REAL, level TEXT, category TEXT, proto TEXT, service TEXT, state TEXT,
  srcip TEXT, dstip TEXT, sbytes REAL, dbytes REAL, sttl REAL, truth TEXT
);
CREATE INDEX IF NOT EXISTS ix_alerts_inc ON alerts(incident_id);
CREATE TABLE IF NOT EXISTS incidents (
  id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, first_ts REAL, last_ts REAL, created REAL,
  category TEXT, service TEXT, proto TEXT, dstip TEXT, srcips TEXT DEFAULT '[]',
  count INTEGER DEFAULT 0, max_score REAL, level TEXT, max_p REAL,
  reasons TEXT DEFAULT '[]', sample TEXT DEFAULT '{}', telegram TEXT, siem TEXT,
  status TEXT DEFAULT 'open', verdict TEXT DEFAULT '', analyst TEXT DEFAULT '', note TEXT DEFAULT '',
  truth_true INTEGER DEFAULT 0, truth_false INTEGER DEFAULT 0, batch_id INTEGER, variant TEXT, thr REAL
);
CREATE INDEX IF NOT EXISTS ix_inc_key ON incidents(key, last_ts);
CREATE TABLE IF NOT EXISTS outbox (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, channel TEXT, incident_id INTEGER, level TEXT,
  text TEXT, status TEXT, error TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS batches (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, source TEXT, name TEXT, summary TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS calibrations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, source TEXT, n INTEGER, percentile REAL, thr REAL, variant TEXT
);
"""

DEFAULTS = {"thr_mode": "cal95", "variant": "full", "window": "60", "telegram_token": "", "telegram_chat": "",
            "siem_path": "", "smtp_host": "", "smtp_port": "587", "smtp_user": "", "smtp_pass": "", "smtp_to": "",
            "soar": "on", "thr_local": "", "thr_local_nottl": ""}


class Storage:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)
            for k, v in DEFAULTS.items():
                c.execute("INSERT OR IGNORE INTO settings VALUES (?,?)", (k, v))

    @contextmanager
    def conn(self):
        with _lock:
            c = sqlite3.connect(self.path, timeout=30)
            c.row_factory = sqlite3.Row
            try:
                yield c
                c.commit()
            finally:
                c.close()

    # ---------------------------------------------------------------- sozlamalar
    def settings(self) -> dict:
        with self.conn() as c:
            return {r["key"]: r["value"] for r in c.execute("SELECT * FROM settings")}

    def get(self, key: str, default: str = "") -> str:
        with self.conn() as c:
            r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    def set(self, key: str, value) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, str(value)))

    # ---------------------------------------------------------------- tahlillar
    def new_batch(self, source: str, name: str) -> int:
        with self.conn() as c:
            return c.execute("INSERT INTO batches (created, source, name) VALUES (?,?,?)", (time.time(), source, name)).lastrowid

    def finish_batch(self, bid: int, summary: dict) -> None:
        with self.conn() as c:
            c.execute("UPDATE batches SET summary=? WHERE id=?", (json.dumps(summary, ensure_ascii=False), bid))

    def batches(self, limit=50) -> list[dict]:
        with self.conn() as c:
            rows = [dict(r) for r in c.execute("SELECT * FROM batches ORDER BY id DESC LIMIT ?", (limit,))]
        for r in rows:
            r["summary"] = json.loads(r["summary"] or "{}")
        return rows

    def batch(self, bid: int) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM batches WHERE id=?", (bid,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["summary"] = json.loads(d["summary"] or "{}")
        return d

    # ---------------------------------------------------------------- ogohlantirishlar va insidentlar
    def add_alerts(self, rows: list[tuple]) -> None:
        with self.conn() as c:
            c.executemany("INSERT INTO alerts (ts,batch_id,incident_id,p,ae,ae_flag,score,level,category,proto,service,state,"
                          "srcip,dstip,sbytes,dbytes,sttl,truth) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    def recent_incidents(self, since: float) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT id, key, last_ts FROM incidents WHERE last_ts>=?", (since,))]

    def create_incident(self, d: dict) -> int:
        cols = list(d)
        with self.conn() as c:
            return c.execute(f"INSERT INTO incidents ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                             [json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for v in d.values()]).lastrowid

    def update_incident_stats(self, iid: int, last_ts: float, add: int, max_score: float, level: str, max_p: float,
                              t_true: int, t_false: int, srcips: list[str]) -> None:
        with self.conn() as c:
            r = c.execute("SELECT srcips, max_score FROM incidents WHERE id=?", (iid,)).fetchone()
            ips = sorted(set(json.loads(r["srcips"] or "[]")) | set(srcips))[:50]
            up = max_score > (r["max_score"] or 0)
            c.execute("UPDATE incidents SET last_ts=MAX(last_ts, ?), count=count+?, max_score=MAX(max_score, ?), "
                      "level=CASE WHEN ? THEN ? ELSE level END, max_p=MAX(max_p, ?), truth_true=truth_true+?, "
                      "truth_false=truth_false+?, srcips=? WHERE id=?",
                      (last_ts, add, max_score, int(up), level, max_p, t_true, t_false, json.dumps(ips), iid))

    def set_incident_text(self, iid: int, telegram: str, siem: dict) -> None:
        with self.conn() as c:
            c.execute("UPDATE incidents SET telegram=?, siem=? WHERE id=?", (telegram, json.dumps(siem, ensure_ascii=False), iid))

    def incidents(self, level: str = "", status: str = "", verdict: str = "", limit: int = 300, offset: int = 0) -> list[dict]:
        sql, a = "SELECT * FROM incidents WHERE 1=1", []
        if level:
            sql += " AND level=?"
            a.append(level)
        if status:
            sql += " AND status=?"
            a.append(status)
        if verdict:
            sql += " AND verdict=?"
            a.append(verdict)
        sql += " ORDER BY CASE level WHEN 'P1' THEN 0 WHEN 'P2' THEN 1 WHEN 'P3' THEN 2 ELSE 3 END, last_ts DESC LIMIT ? OFFSET ?"
        a += [limit, offset]
        with self.conn() as c:
            return [self._inc(r) for r in c.execute(sql, a)]

    def incident(self, iid: int) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM incidents WHERE id=?", (iid,)).fetchone()
        return self._inc(r) if r else None

    @staticmethod
    def _inc(r) -> dict:
        d = dict(r)
        for k in ("reasons", "srcips"):
            d[k] = json.loads(d[k] or "[]")
        d["sample"] = json.loads(d["sample"] or "{}")
        d["siem"] = json.loads(d["siem"]) if d.get("siem") else {}
        return d

    def incident_alerts(self, iid: int, limit: int = 200) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM alerts WHERE incident_id=? ORDER BY ts LIMIT ?", (iid, limit))]

    def set_verdict(self, iid: int, verdict: str, analyst: str, note: str, status: str) -> None:
        with self.conn() as c:
            c.execute("UPDATE incidents SET verdict=?, analyst=?, note=?, status=? WHERE id=?",
                      (verdict, analyst[:80], note[:2000], status, iid))

    # ---------------------------------------------------------------- xabarlar
    def add_outbox(self, channel: str, incident_id: int | None, level: str, text: str, status: str, error: str = "") -> int:
        with self.conn() as c:
            return c.execute("INSERT INTO outbox (ts,channel,incident_id,level,text,status,error) VALUES (?,?,?,?,?,?,?)",
                             (time.time(), channel, incident_id, level, text, status, error[:500])).lastrowid

    def outbox(self, channel: str = "", status: str = "", limit: int = 200) -> list[dict]:
        sql, a = "SELECT * FROM outbox WHERE 1=1", []
        if channel:
            sql += " AND channel=?"
            a.append(channel)
        if status:
            sql += " AND status=?"
            a.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        a.append(limit)
        with self.conn() as c:
            return [dict(r) for r in c.execute(sql, a)]

    def outbox_for(self, incident_id: int) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM outbox WHERE incident_id=? ORDER BY id", (incident_id,))]

    def mark_outbox(self, ids: list[int], status: str) -> None:
        with self.conn() as c:
            c.executemany("UPDATE outbox SET status=? WHERE id=?", [(status, i) for i in ids])

    # ---------------------------------------------------------------- kalibrlash
    def add_calibration(self, source: str, n: int, pct: float, thr: float, variant: str) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO calibrations (created,source,n,percentile,thr,variant) VALUES (?,?,?,?,?,?)",
                      (time.time(), source, n, pct, thr, variant))

    def calibrations(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM calibrations ORDER BY id DESC LIMIT 20")]

    # ---------------------------------------------------------------- statistika
    def stats(self) -> dict:
        with self.conn() as c:
            one = lambda s, *a: c.execute(s, a).fetchone()[0]  # noqa: E731
            out = {
                "flows": sum(b["summary"].get("flows", 0) for b in self.batches(1000)),
                "alerts": one("SELECT COUNT(*) FROM alerts"),
                "alert_levels": {r["level"]: r["n"] for r in c.execute("SELECT level, COUNT(*) n FROM alerts GROUP BY level")},
                "incidents": one("SELECT COUNT(*) FROM incidents"),
                "open": {r["level"]: r["n"] for r in c.execute(
                    "SELECT level, COUNT(*) n FROM incidents WHERE status='open' GROUP BY level")},
                "categories": [dict(r) for r in c.execute(
                    "SELECT category, COUNT(*) n, SUM(count) flows FROM incidents GROUP BY category ORDER BY flows DESC")],
                "verdicts": {r["verdict"]: r["n"] for r in c.execute(
                    "SELECT verdict, COUNT(*) n FROM incidents WHERE verdict<>'' GROUP BY verdict")},
                "truth": dict(c.execute("SELECT COALESCE(SUM(truth_true),0) t, COALESCE(SUM(truth_false),0) f FROM incidents").fetchone()),
                "outbox": {f"{r['channel']}:{r['status']}": r["n"] for r in c.execute(
                    "SELECT channel, status, COUNT(*) n FROM outbox GROUP BY channel, status")},
                "timeline": [dict(r) for r in c.execute(
                    "SELECT CAST(ts/60 AS INTEGER)*60 bucket, level, COUNT(*) n FROM alerts GROUP BY bucket, level ORDER BY bucket")],
            }
        return out

    def reset(self) -> None:
        with self.conn() as c:
            for t in ("alerts", "incidents", "outbox", "batches", "calibrations"):
                c.execute(f"DELETE FROM {t}")
            c.execute("DELETE FROM sqlite_sequence")

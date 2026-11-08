"""5-qatlam: ogohlantirishlarni yetkazish (diplom, D-ilova).

  SIEM (Wazuh)   — barcha darajalar: JSON qatorlari fayliga ({"ai_alert": {...}}), Wazuh localfile orqali o‘qiydi
  Telegram       — P1 va P2 darhol (Bot API, 429 cheklovida kutib qayta urinish)
  E-pochta       — P3 va P4 soatlik dayjest (SMTP, ixtiyoriy)
  SOAR           — P1 uchun avtomatik chora yozuvi (playbook)
Token va parollar ma’lumotlar bazasida yoki muhit o‘zgaruvchilarida saqlanadi, kodda emas.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import time
import urllib.error
import urllib.request
from email.mime.text import MIMEText

from .features import CLASS_UZ, RECOMMEND
from .priority import LEVEL_NAME


def telegram_text(inc: dict) -> str:
    """B-ilova, telegram_text(): tahlilchi bir qarashda turi, prioriteti, ishonch va sabablarni ko‘radi."""
    r = "\n".join(f" • {x['name']} = {x['value']} (+{x['shap']})" for x in inc.get("reasons", []))
    cat = inc["category"]
    src = f"\nManba: {', '.join(inc['srcips'][:3])}" if inc.get("srcips") else ""
    dst = f" → {inc['dstip']}" if inc.get("dstip") else ""
    return (f"[{inc['level']} – {LEVEL_NAME.get(inc['level'], '')}] {cat if cat != 'Anomaliya' else 'Noma’lum anomaliya'}\n"
            f"Ehtimollik: {inc['max_p']:.1%} · Bal: {inc['max_score']}\n"
            f"Xizmat/protokol: {inc['service']}/{inc['proto']}{dst}{src}\n"
            f"Asosiy sabablar:\n{r}\n"
            f"Shu turdagi hodisalar ({inc.get('window', 60)} s): {inc.get('count', 1)}\n"
            f"Tavsiya: {RECOMMEND.get(cat, 'oqimni qo‘lda tekshiring')}")


def siem_record(inc: dict) -> dict:
    return {"id": inc.get("id"), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(inc["first_ts"])),
            "priority": inc["level"], "level": LEVEL_NAME.get(inc["level"], ""), "category": inc["category"],
            "category_uz": CLASS_UZ.get(inc["category"], "Noma’lum anomaliya"), "probability": round(inc["max_p"], 4),
            "score": inc["max_score"], "service": inc["service"], "proto": inc["proto"], "dstip": inc.get("dstip") or None,
            "srcips": inc.get("srcips", []), "count": inc.get("count", 1),
            "reasons": [{"feature": x["feature"], "value": x["value"], "shap": x["shap"]} for x in inc.get("reasons", [])],
            "recommendation": RECOMMEND.get(inc["category"], ""), "source": "SentinelAI"}


class Channels:
    def __init__(self, store, default_siem: str):
        self.store = store
        self.default_siem = default_siem

    def cfg(self) -> dict:
        s = self.store.settings()
        s["telegram_token"] = s.get("telegram_token") or os.environ.get("SOC_BOT_TOKEN", "")
        s["telegram_chat"] = s.get("telegram_chat") or os.environ.get("SOC_CHAT_ID", "")
        s["siem_path"] = s.get("siem_path") or self.default_siem
        return s

    # ------------------------------------------------------------ SIEM
    def to_siem(self, rec: dict, cfg: dict) -> tuple[bool, str]:
        path = cfg["siem_path"]
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ai_alert": rec}, ensure_ascii=False) + "\n")
            return True, ""
        except OSError as e:
            return False, str(e)

    # ------------------------------------------------------------ Telegram
    def to_telegram(self, text: str, cfg: dict, retries: int = 3) -> tuple[str, str]:
        token, chat = cfg["telegram_token"], cfg["telegram_chat"]
        if not token or not chat:
            return "preview", "Telegram sozlanmagan (token yoki chat_id yo‘q) — xabar faqat konsolda ko‘rsatildi"
        api = f"https://api.telegram.org/bot{token}/sendMessage"
        body = json.dumps({"chat_id": chat, "text": text, "disable_web_page_preview": True}).encode()
        err = ""
        for k in range(retries):
            try:
                req = urllib.request.Request(api, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as r:
                    if r.status == 200:
                        return "sent", ""
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}"
                if e.code == 429:                       # cheklov: kutib qayta urinish
                    try:
                        wait = json.loads(e.read().decode()).get("parameters", {}).get("retry_after", 2)
                    except (ValueError, OSError):
                        wait = 2
                    time.sleep(min(float(wait), 10))
                    continue
                if e.code in (400, 401, 403, 404):
                    break
            except (urllib.error.URLError, OSError) as e:
                err = str(getattr(e, "reason", e))
            logging.warning("Telegram xatosi: %s", err)
            time.sleep(min(2 ** k, 4))
        return "failed", err or "noma’lum xato"

    # ------------------------------------------------------------ e-pochta dayjesti
    def digest_text(self, items: list[dict]) -> str:
        lines = [f"SentinelAI — ogohlantirishlar dayjesti ({time.strftime('%Y-%m-%d %H:%M')})",
                 f"Jami: {len(items)} ta P3/P4 insident", ""]
        for o in items:
            lines.append(f"#{o['incident_id']} [{o['level']}] " + o["text"].split("\n", 1)[0])
        return "\n".join(lines)

    def send_digest(self) -> tuple[str, str, int]:
        cfg = self.cfg()
        items = self.store.outbox(channel="digest", status="queued", limit=1000)
        if not items:
            return "empty", "Dayjest navbati bo‘sh", 0
        text = self.digest_text(list(reversed(items)))
        status, err = "preview", "SMTP sozlanmagan — dayjest faqat konsolda ko‘rsatildi"
        if cfg.get("smtp_host") and cfg.get("smtp_to"):
            try:
                msg = MIMEText(text, "plain", "utf-8")
                msg["Subject"] = f"SentinelAI dayjest: {len(items)} ta ogohlantirish"
                msg["From"] = cfg.get("smtp_user") or "sentinel@localhost"
                msg["To"] = cfg["smtp_to"]
                with smtplib.SMTP(cfg["smtp_host"], int(cfg.get("smtp_port") or 587), timeout=10) as s:
                    s.starttls()
                    if cfg.get("smtp_user"):
                        s.login(cfg["smtp_user"], cfg.get("smtp_pass", ""))
                    s.send_message(msg)
                status, err = "sent", ""
            except (OSError, smtplib.SMTPException) as e:
                status, err = "failed", str(e)
        self.store.mark_outbox([o["id"] for o in items], "digested")
        self.store.add_outbox("email", None, "P3/P4", text, status, err)
        return status, err, len(items)

    # ------------------------------------------------------------ yo‘naltirish (D-ilova, dispatch())
    def dispatch(self, inc: dict, text: str, rec: dict, cfg: dict | None = None, telegram: bool = True) -> dict:
        cfg = cfg or self.cfg()
        out = {}
        ok, err = self.to_siem(rec, cfg)                                   # barcha darajalar SIEM ga
        self.store.add_outbox("siem", inc["id"], inc["level"], json.dumps(rec, ensure_ascii=False),
                              "sent" if ok else "failed", err)
        out["siem"] = ok
        if inc["level"] in ("P1", "P2"):                                   # darhol
            st, err = self.to_telegram(text, cfg) if telegram else ("preview", "ommaviy tahlilda Telegram o‘chirilgan")
            self.store.add_outbox("telegram", inc["id"], inc["level"], text, st, err)
            out["telegram"] = st
        else:                                                              # P3–P4: soatlik dayjest
            self.store.add_outbox("digest", inc["id"], inc["level"], text, "queued")
            out["digest"] = True
        if inc["level"] == "P1" and cfg.get("soar") == "on":
            playbook = f"SOAR playbook «{inc['category']}»: {RECOMMEND.get(inc['category'], '')}"
            if inc.get("srcips"):
                playbook += f"; manba bloklash tavsiyasi: {', '.join(inc['srcips'][:3])}"
            self.store.add_outbox("soar", inc["id"], inc["level"], playbook, "done")
            out["soar"] = True
        return out

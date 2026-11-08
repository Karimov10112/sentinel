"""Ogohlantirish prioriteti (diplom, (3.2) formula) va darajalar."""
from __future__ import annotations

from .features import SEV

LEVELS = [(0.85, "P1", "KRITIK"), (0.60, "P2", "YUQORI"), (0.40, "P3", "O‘RTA"), (0.25, "P4", "PAST")]
LEVEL_NAME = {c: n for _, c, n in LEVELS}
CHANNEL = {"P1": "Telegram darhol + SOAR", "P2": "Telegram darhol", "P3": "e-pochta dayjesti (soatlik)", "P4": "faqat SIEM"}
ORDER = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}


def score(p: float, cat: str, ae_flag: bool, thr: float) -> float:
    """S = p · (0,5 + 0,5 · w_tur) + 0,25 · [e(x) > θ_AE] · [p < θ]   (3.2)

    Birinchi had faqat chegaradan o‘tgan oqimlar uchun hisoblanadi; ikkinchi had XGBoost o‘tkazib yuborgan, lekin
    autoenkoder anomal deb topgan oqimlarga past prioritetli (P4) ogohlantirish beradi."""
    s = p * (0.5 + 0.5 * SEV.get(cat, 0.3)) if p >= thr else 0.0
    if ae_flag and p < thr:
        s += 0.25
    return round(s, 4)


def level(s: float) -> str:
    for t, code, _ in LEVELS:
        if s >= t:
            return code
    return ""

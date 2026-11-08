"""2-qatlam: oqim belgilarini tayyorlash (diplom, 2.2-band, (2.1) formula).

UNSW-NB15 oqim yozuvi (Argus + Zeek belgilari) → z = (ln(1 + x) − μ) / σ  (39 ta sonli belgi) + one-hot (proto, service, state)
→ 194 o‘lchamli vektor. Parametrlar JSON’da saqlanadi, shuning uchun scikit-learn versiyasiga bog‘liq emas.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

CAT = ["proto", "service", "state"]
NUM = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
       "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean",
       "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
       "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
       "is_sm_ips_ports"]
TTL = ["sttl", "dttl", "ct_state_ttl"]                       # topologiyaga bog‘liq belgilar (3.5-band)
COLUMNS = NUM + CAT
CLASSES = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]
CLASS_UZ = {"Normal": "Normal trafik", "Generic": "Generic (kriptografik primitivlarga hujum)",
            "Exploits": "Exploits (zaiflikdan foydalanish)", "Fuzzers": "Fuzzers (tasodifiy ma’lumot bilan sinash)",
            "DoS": "DoS (xizmat ko‘rsatishni rad etish)", "Reconnaissance": "Reconnaissance (razvedka, skanerlash)",
            "Analysis": "Analysis (port/HTML tahlil, spam)", "Backdoor": "Backdoor (yashirin kirish)",
            "Shellcode": "Shellcode (zararli kod in’eksiyasi)", "Worms": "Worms (o‘z-o‘zidan tarqaluvchi dastur)"}
# (3.2) dagi hujum turining og‘irligi w_tur
SEV = {"Backdoor": 1.0, "Worms": 1.0, "Shellcode": 0.95, "Exploits": 0.9, "DoS": 0.7, "Generic": 0.6, "Analysis": 0.5,
       "Fuzzers": 0.4, "Reconnaissance": 0.3, "Normal": 0.3}
GROUP = {"Exploits": "Zaiflikdan foydalanish", "DoS": "Zaiflikdan foydalanish", "Backdoor": "Zaiflikdan foydalanish",
         "Shellcode": "Zararli kod", "Worms": "Zararli kod", "Reconnaissance": "Razvedka", "Analysis": "Razvedka",
         "Fuzzers": "Ilova darajasidagi hujum", "Generic": "Kriptografik hujum", "Normal": "—"}
RECOMMEND = {"Exploits": "zaiflik nishonidagi xostni tekshiring, IPS qoidasini yoqing",
             "DoS": "trafik cheklovini (rate-limit) yoqing",
             "Backdoor": "xostni izolyatsiya qiling, EDR skan",
             "Shellcode": "xostni izolyatsiya qiling, xotira tahlili",
             "Worms": "segmentni ajrating, tarqalishni to‘xtating",
             "Reconnaissance": "manba IP ni kuzatuvga oling",
             "Fuzzers": "ilova jurnallarini tekshiring",
             "Generic": "kriptografik xizmat jurnallarini tekshiring",
             "Analysis": "pochta va veb-filtrlarni tekshiring",
             "Normal": "anomal oqimni qo‘lda tekshiring"}
MITRE = {"Exploits": "T1190", "DoS": "T1499", "Backdoor": "T1505", "Shellcode": "T1203", "Worms": "T1210",
         "Reconnaissance": "T1046", "Analysis": "T1595", "Fuzzers": "T1190", "Generic": "T1600", "Normal": ""}
UZ = {"sttl": "manba TTL qiymati", "dttl": "javob TTL qiymati", "ct_dst_sport_ltm": "bir manba portidan bir manzilga ulanishlar",
      "ct_state_ttl": "holat va TTL kombinatsiyasi", "sbytes": "manbadan yuborilgan baytlar", "dbytes": "javob baytlari",
      "smean": "manba paketining o‘rtacha hajmi", "dmean": "javob paketining o‘rtacha hajmi",
      "ct_srv_dst": "bir xizmatga bir manzilga ulanishlar", "ct_srv_src": "bir xizmatga bir manbadan ulanishlar",
      "rate": "paket tezligi", "sload": "manba yuklamasi (bit/s)", "dload": "javob yuklamasi (bit/s)",
      "dur": "ulanish davomiyligi", "ct_dst_src_ltm": "manba-manzil ulanishlari soni", "spkts": "manba paketlari soni",
      "dpkts": "javob paketlari soni", "sinpkt": "manba paketlararo vaqt", "dinpkt": "javob paketlararo vaqt",
      "sjit": "manba jitteri", "djit": "javob jitteri", "swin": "manba TCP oynasi", "dwin": "javob TCP oynasi",
      "stcpb": "manba TCP ketma-ketlik raqami", "dtcpb": "javob TCP ketma-ketlik raqami", "tcprtt": "TCP aloqa vaqti (RTT)",
      "synack": "SYN–SYN/ACK vaqti", "ackdat": "SYN/ACK–ACK vaqti", "trans_depth": "HTTP so‘rovlar chuqurligi",
      "response_body_len": "HTTP javob hajmi", "ct_dst_ltm": "bir manzilga ulanishlar", "ct_src_ltm": "bir manbadan ulanishlar",
      "ct_src_dport_ltm": "bir manbadan bir portga ulanishlar", "is_ftp_login": "FTP login", "ct_ftp_cmd": "FTP buyruqlari",
      "ct_flw_http_mthd": "HTTP metodlari soni", "sloss": "manba paket yo‘qotishi", "dloss": "javob paket yo‘qotishi",
      "is_sm_ips_ports": "manba va manzil IP/port bir xil", "proto": "protokol", "service": "xizmat", "state": "holat"}


class Prep:
    """log1p + StandardScaler + OneHotEncoder(handle_unknown='ignore') ekvivalenti (A-ilova, X() funksiyasi)."""

    def __init__(self, num: list[str] | None = None, cat: list[str] | None = None):
        self.num = list(num or NUM)
        self.cat = list(cat or CAT)
        self.mu = self.sd = None
        self.cats: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame) -> "Prep":
        z = np.log1p(df[self.num].clip(lower=0).astype(float).values)
        self.mu, self.sd = z.mean(0), z.std(0)
        self.sd[self.sd == 0] = 1.0
        self.cats = {c: sorted(df[c].astype(str).unique().tolist()) for c in self.cat}
        return self

    @property
    def names(self) -> list[str]:
        return self.num + [f"{c}_{v}" for c in self.cat for v in self.cats[c]]

    @property
    def dim(self) -> int:
        return len(self.num) + sum(len(v) for v in self.cats.values())

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        z = (np.log1p(np.clip(df[self.num].astype(float).values, 0, None)) - self.mu) / self.sd
        parts = [z.astype(np.float32)]
        for c in self.cat:
            idx = {v: i for i, v in enumerate(self.cats[c])}
            oh = np.zeros((len(df), len(idx)), np.float32)
            col = df[c].astype(str).map(idx)
            ok = col.notna().values
            oh[np.where(ok)[0], col[ok].astype(int).values] = 1.0
            parts.append(oh)
        return np.hstack(parts)

    def to_json(self) -> dict:
        return {"num": self.num, "cat": self.cat, "mu": self.mu.tolist(), "sd": self.sd.tolist(), "cats": self.cats}

    @classmethod
    def from_json(cls, d: dict) -> "Prep":
        p = cls(d["num"], d["cat"])
        p.mu, p.sd, p.cats = np.array(d["mu"]), np.array(d["sd"]), d["cats"]
        return p

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f)

    @classmethod
    def load(cls, path: str) -> "Prep":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Kirish oqimlarini tozalash: etishmayotgan ustunlar 0 / '-', matnli kategoriyalar."""
    df = df.copy()
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    for c in NUM:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in CAT:
        if c not in df.columns:
            df[c] = "-"
        df[c] = df[c].astype(str).str.strip()
    return df

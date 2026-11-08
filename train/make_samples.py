"""samples/demo_with_ips.csv — vaqt va IP manzillar qo‘shilgan namunaviy oqimlar (insidentlarni manzil bo‘yicha guruhlash namoyishi)."""
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(ROOT, "samples", "stream_20000.csv")).sample(3000, random_state=5).reset_index(drop=True)
    rs = np.random.RandomState(5)
    attackers = {"Exploits": ["185.220.101.47", "45.155.205.233"], "DoS": ["91.240.118.172"], "Reconnaissance": ["45.155.205.233"],
                 "Fuzzers": ["103.152.118.20"], "Generic": ["194.26.29.113"], "Backdoor": ["5.188.206.14"],
                 "Analysis": ["80.94.95.115"], "Shellcode": ["185.220.101.47"], "Worms": ["10.0.10.41"]}
    servers = ["10.0.2.10", "10.0.1.10", "10.0.1.20", "10.0.2.20"]
    cat = df.attack_cat.astype(str).str.strip()
    src, dst = [], []
    for c in cat:
        if c == "Normal":
            src.append(f"10.0.10.{rs.randint(11, 61)}")
            dst.append(rs.choice(servers + ["142.250.74.46", "13.107.42.14"]))
        else:
            src.append(rs.choice(attackers.get(c, ["203.0.113.5"])))
            dst.append(rs.choice(servers[:2]))
    df.insert(0, "dstip", dst)
    df.insert(0, "srcip", src)
    df.insert(0, "ts", (time.time() - 900 + np.arange(len(df)) * 0.3).round(3))
    df.to_csv(os.path.join(ROOT, "samples", "demo_with_ips.csv"), index=False)
    print("samples/demo_with_ips.csv:", len(df))

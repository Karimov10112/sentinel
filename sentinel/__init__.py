"""SentinelAI — sun’iy intellekt asosida kiberhujumlarni aniqlash va ogohlantirish tizimi.

Besh qatlamli arxitektura (diplom, 3.1-band):
  1. Sensor va oqim generatori (Zeek / Argus) — UNSW-NB15 formatidagi oqim yozuvlari
  2. Belgilarni tayyorlash — log(1+x) + standartlash + one-hot, 194 o‘lchamli vektor (features.py)
  3. SI dvigateli — XGBoost (ehtimollik, hujum turi), autoenkoder (anomaliya), DNN (zaxira) (models.py, nn.py)
  4. Ogohlantirish menejeri — kalibrlangan chegara, prioritet (3.2), SHAP sabablari, insidentlarga guruhlash (alerting.py)
  5. Yetkazish va javob — SIEM (Wazuh JSON), Telegram, e-pochta dayjesti, SOAR (channels.py)
"""
__version__ = "1.0.0"

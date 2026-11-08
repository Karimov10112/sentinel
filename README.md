# SentinelAI — sun’iy intellekt asosida kiberhujumlarni aniqlash va ogohlantirish tizimi

Diplom loyihasining dasturiy qismi. Tizim diplomning 3-bobidagi **besh qatlamli arxitekturani** amalga oshiradi:

| Qatlam | Modul | Kirish → chiqish | Fayl |
|---|---|---|---|
| 1 | Sensor, oqim generatori | paketlar → oqim yozuvlari (Zeek/Argus, UNSW-NB15 formati) | API, fayl yuklash |
| 2 | Belgilarni tayyorlash | oqim → 194 o‘lchamli vektor, z = (ln(1+x) − μ)/σ + one-hot | `sentinel/features.py` |
| 3 | SI dvigateli | vektor → ehtimollik p (XGBoost), hujum turi (10 sinf), anomaliya e(x) (autoenkoder), DNN — zaxira | `sentinel/models.py`, `sentinel/nn.py` |
| 4 | Ogohlantirish menejeri | kalibrlangan chegara, prioritet (3.2), SHAP sabablari, 60 s oynada insidentlarga guruhlash | `sentinel/alerting.py` |
| 5 | Yetkazish va javob | Wazuh SIEM (JSON), Telegram (P1–P2 darhol), e-pochta dayjesti (P3–P4), SOAR (P1) | `sentinel/channels.py` |

```
S = p·(0,5 + 0,5·w_tur) + 0,25·[e(x) > θ_AE]·[p < θ]          (3.2)
P1: S ≥ 0,85   P2: 0,60–0,85   P3: 0,40–0,60   P4: 0,25–0,40
C(θ) = 10·FN(θ) + FP(θ)                                        (3.1)
```

## Tez ishga tushirish

**Windows:** `run_windows.bat` faylini ikki marta bosing (Python 3.10+ kerak). Brauzerda http://127.0.0.1:5000 ochiladi.

**Linux / macOS:** `./run.sh`, yoki qo‘lda:
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Boshqaruv panelidagi **«▶ 5 000 ta oqimni tahlil qilish»** tugmasi UNSW-NB15 sinov to‘plamidan, kalibrlashga kirmagan oqimlarni
prototip quvuridan o‘tkazadi: insidentlar, SHAP sabablari, Telegram xabarlari va SIEM yozuvlari hosil bo‘ladi.

Modellar portativ formatlarda saqlangan (XGBoost JSON, NumPy npz, JSON), shuning uchun qayta o‘qitish talab etilmaydi va
PyTorch o‘rnatish shart emas.

## Veb-konsol

- **Boshqaruv paneli** — oqimlar, P1–P4 ogohlantirishlar, insidentlar, aniqlik, kanallar holati, hujum turlari.
- **Insidentlar** — prioritet va holat bo‘yicha filtr. Insident sahifasida:
  - taxminiy tur va ATT&CK texnikasi;
  - TreeSHAP sabablari (12 ta belgi);
  - 10 sinf bo‘yicha ehtimolliklar;
  - Telegram xabari ko‘rinishi va SIEM JSON yozuvi;
  - yetkazish tarixi;
  - tahlilchi bahosi («Haqiqiy hujum» / «Soxta ogohlantirish»).
- **Oqimni tekshirish** — bitta oqim uchun:
  - XGBoost natijasi, DNN (zaxira) bilan solishtirish;
  - autoenkoder xatosi;
  - prioritet hisobi va SHAP grafigi.
- **Oqimlar tahlili** — UNSW-NB15 sinov oqimi (1 000–20 000) yoki o‘z CSV/JSON faylingiz. Natija 3.8-jadval shaklida chiqadi.
- **Kalibrlash** — chegara rejimi:
  - asl 0,5;
  - xatolar narxi bo‘yicha;
  - 95- yoki 99-persentil;
  - mahalliy kalibrlash.

  Shu yerda model tanlanadi (barcha belgilar yoki TTL belgilarisiz) va guruhlash oynasi beriladi. 3.6-jadval, chegara egri chizig‘i va mahalliy namuna hajmining ta’siri ham shu sahifada.
- **Kanallar** — Telegram bot tokeni va chat ID, SIEM fayli, SMTP, sinov xabari, dayjest yuborish, xabarlar jurnali.
- **Monitoring** — har bir tahlil uchun:
  - taqsimot siljishi (PSI);
  - `sttl = 254` ulushi;
  - 10 000 oqimga ogohlantirishlar soni;
  - Precision (tahlilchi baholari bo‘yicha).

  Bu yerdan yorliqlarni qayta o‘qitish uchun eksport qilish mumkin.
- **SI modellar** — diplomdagi 3.3–3.8-jadvallar, ROC, autoenkoder xatosi taqsimoti, chalkashlik matritsasi, SHAP (global va beeswarm), TTL tajribasi.
- **API**, **Tizim haqida**.

## Tajriba natijalari (UNSW-NB15, `python train/train_models.py`)

O‘qitish 157 806, validatsiya 17 535, sinov 82 332 oqim, 194 belgi. Binar tasnif (3.3-jadval):

| Model | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0,838 | 0,783 | 0,978 | 0,869 | 33,3% | 0,967 |
| Random Forest | 0,867 | 0,811 | 0,989 | 0,891 | 28,2% | 0,983 |
| **XGBoost** | **0,870** | **0,819** | 0,982 | **0,893** | 26,6% | 0,983 |
| DNN (MLP) | 0,875 | 0,820 | 0,989 | 0,897 | 26,6% | 0,984 |
| Autoenkoder | 0,849 | 0,926 | 0,788 | 0,851 | 7,7% | 0,907 |
| Isolation Forest | 0,611 | 0,831 | 0,368 | 0,511 | 9,2% | 0,807 |
| Gibrid (XGBoost + AE) | 0,856 | 0,800 | 0,985 | 0,883 | 30,2% | — |

**Diplomdagi raqamlar bilan aynan mos chiqqanlar:**
- 3.3-jadval: LR, RF va XGBoost.
- Ko‘p sinfli XGBoost: Accuracy 0,732, makro-F1 0,533.
- SHAP: sttl hissasi 2,88, keyingi belgi 0,88.
- Taqsimot siljishi: sttl = 254 ulushi 20,1% → 40,7%; soxta ogohlantirishlarning 87,6% i shu oqimlarda.
- 3.5-jadval (TTL belgilarini olib tashlash).
- Xatolar narxi bo‘yicha chegara θ = 0,09 (sinovda Recall 0,999, FPR 41,6%).
- 3.6-jadvalning A va B qatorlari: soxta ogohlantirishlar 7 802 → 1 352 → 214.
- 3.7-jadvaldagi P1 qatori: 14 039 / 13 677 / 362.

Prototip (3.8-jadval, kalibrlashga kirmagan 20 000 oqim, θ = 0,928):

| Ko‘rsatkich | Diplomda | Loyihada |
|---|---|---|
| Precision | 96,8% | 96,8% |
| Recall | 90,7% | 91,1% |
| Insidentlar (60 s oyna) | 301 | 291 |
| P1 / P2 aniqligi | 98,7% / 95,8% | 98,6% / 96,0% |

**Diplomdan farqlar (himoyada aytish uchun):**
- **DNN va autoenkoder** PyTorch o‘rniga NumPy’da yozilgan. Arxitektura, optimizator va erta to‘xtatish xuddi A-ilovadagidek. Natijalar biroz farq qiladi:
  - DNN F1: 0,897 (diplomda 0,888);
  - autoenkoder FPR: 7,7% (diplomda 8,5%);
  - ko‘p sinfli DNN: 0,698 / 0,464 (diplomda 0,700 / 0,466).
- **Isolation Forest** F1 = 0,511 (diplomda 0,454). Sabab — sklearn versiyasi va chegara hisoblash tafsiloti.
- **3.6-jadvalning «C» qatori** (qayta o‘qitish): FPR 13,3% (diplomda 14,1%).
- **P4 darajasi:** 2 837 ta ogohlantirish (diplomda 5 465). Bu yerdagi P4 — faqat autoenkoder anomaliyalari, shuning uchun soni autoenkoder natijasiga bog‘liq.
- **Guruhlash:** 60 s oynada 568 ta insident (diplomda 558).

## Real tarmoqqa joriy etish

1. **Oqim generatori.** U UNSW-NB15 belgilarini (Argus + Zeek, 2.2-jadval) CSV yoki JSON ko‘rinishida beradi va ularni `POST /api/v1/flows` ga yuboradi. Ixtiyoriy ustunlar: `ts`, `srcip`, `dstip`.
2. **Pilot (2–4 hafta).** Hujumsiz deb tekshirilgan mahalliy normal oqimlar yig‘iladi va «Kalibrlash» sahifasida 95-persentil bo‘yicha chegara o‘rnatiladi. Mahalliy kalibrlashsiz FPR ≈ 26%, kalibrlashdan keyin ≈ 4,6%.
3. **Kanallar.** Telegram bot (@BotFather) va SOC guruhining chat ID si kiritiladi. Wazuh’ga `integrations/wazuh/local_rules.xml` qo‘shiladi, `ossec.conf` ga localfile bloki yoziladi (Kanallar sahifasida ko‘rsatilgan).
4. **Monitoring.** Tahlilchilar insidentlarni baholaydi; PSI > 0,25 bo‘lsa chegara qayta kalibrlanadi.
5. **Har oy qayta o‘qitish:**
   ```
   python train/adapt.py --local local_normal.csv --labels tahlilchi_yorliqlari.csv
   ```
   Eski model avtomatik ravishda `models/backup/` ga saqlanadi.

Boshqa tarmoqqa ko‘chirishda **TTL belgilarisiz model** tavsiya etiladi (3.5-band). Kalibrlash sahifasida bitta tugma bilan tanlanadi.

## Loyiha tuzilmasi

```
app.py                       Flask SOC konsoli va REST API
sentinel/
  features.py                belgilar, (2.1) tayyorlash, hujum turlari, w_tur, tavsiyalar, belgilar nomlari (o‘zbekcha)
  nn.py                      NumPy’da DNN va autoenkoder (Linear, BatchNorm, ReLU, Dropout, Adam, erta to‘xtatish)
  models.py                  SI dvigateli: XGBoost (binar, 10 sinf), autoenkoder, DNN, TreeSHAP
  priority.py                (3.2) formula va P1–P4
  alerting.py                ogohlantirish xizmati (B-ilova): paketlar, chegara, guruhlash, siljish (PSI)
  channels.py                SIEM, Telegram, e-pochta dayjesti, SOAR (D-ilova)
  storage.py                 SQLite
train/
  train_models.py            barcha tajribalar (A- va C-ilovalar) → models/
  adapt.py                   mahalliy trafik bilan qayta o‘qitish
  make_reference.py          siljish monitoringi uchun etalon
  make_samples.py            IP va vaqtli namunaviy fayl
integrations/                wazuh/local_rules.xml, telegram_bot_test.py
samples/                     stream_20000.csv, pilot_normal.csv, demo_with_ips.csv
data/unsw/                   UNSW-NB15 o‘quv va sinov to‘plamlari
tests/test_sentinel.py       21 ta avtomatik test:  python -m pytest -q
```

## Cheklovlar

- Modellar UNSW-NB15 da o‘qitilgan, bu sintetik muhit. Real tarmoqda chegara albatta mahalliy trafik bo‘yicha kalibrlanishi kerak.
- UNSW-NB15 bo‘linmasida vaqt va IP manzillar yo‘q. Guruhlash oqimlar 200 oqim/s tezlikda keladi degan farazga tayanadi, shuning uchun uning natijalari taxminiy.
- Hujum turi «taxminiy tur» sifatida beriladi: DoS, Analysis, Backdoor va Exploits oqim belgilari bo‘yicha o‘zaro chalkashadi.
- Telegram va SMTP faqat foydalanuvchi sozlaganda ishlaydi. Sozlanmagan bo‘lsa, xabarlar konsolda «ko‘rib chiqish» holatida saqlanadi.

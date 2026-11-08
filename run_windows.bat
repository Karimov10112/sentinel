@echo off
chcp 65001 >nul
cd /d %~dp0
if not exist .venv (
  echo [1/3] Virtual muhit yaratilmoqda...
  python -m venv .venv || (echo Python topilmadi. python.org dan Python 3.10+ o'rnating & pause & exit /b 1)
)
call .venv\Scripts\activate
echo [2/3] Kutubxonalar o'rnatilmoqda...
pip install -q -r requirements.txt
python -c "from sentinel.models import Detector; import pandas as pd; d=Detector(); d.score(pd.read_csv('samples/pilot_normal.csv', nrows=5))" 2>nul
if errorlevel 1 (
  echo Modellar yuklanmadi - qayta o'qitiladi, 15-25 daqiqa...
  python train\train_models.py
  python train\make_reference.py
  python train\make_samples.py
)
echo [3/3] SentinelAI ishga tushmoqda: http://127.0.0.1:5000
start "" http://127.0.0.1:5000
python app.py
pause

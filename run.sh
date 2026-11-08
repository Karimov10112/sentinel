#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
if ! python -c "from sentinel.models import Detector; import pandas as pd; d=Detector(); d.score(pd.read_csv('samples/pilot_normal.csv', nrows=5))" 2>/dev/null; then
  echo "Modellar yuklanmadi — qayta o‘qitiladi (15–25 daqiqa)..."
  python train/train_models.py && python train/make_reference.py && python train/make_samples.py
fi
echo "SentinelAI: http://127.0.0.1:5000"
python app.py

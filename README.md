# Energy Consumption Drift Detection & Auto-Retraining ML Pipeline

End-to-end pipeline that predicts **household global active power** (kW) from calendar features, **logs production-like traffic**, **detects statistical drift** vs historical data, and **retrains** when drift exceeds a threshold. Includes a **FastAPI** service, a **Streamlit** dashboard with optional in-browser prediction logging, and **GitHub Actions** CI.

**Dataset:** [Individual household electric power consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (UCI), semicolon-separated text (~2M rows).

---

## Features

| Area | What it does |
|------|----------------|
| **Preprocessing** | Loads raw `.txt`, parses datetimes, keeps `datetime` + `Global_active_power`, writes `data/cleaned_data.csv`. |
| **Training** | `hour`, `day`, `month` → sklearn **LinearRegression** → `model/model.pkl`, metrics, metadata. |
| **Drift** | Normalized mean-shift on power vs reference; writes `model/drift_status.json`. |
| **Retraining** | If drift: refit, save `model_vN.pkl`, promote to `model.pkl`. |
| **API** | `POST /predict` (datetime or hour/day/month); appends rows to `data/new_data.csv`. |
| **Dashboard** | Metrics, drift table, distribution plots, interactive predict + drift check. |
| **CI** | Download data if missing, sampled preprocess, train, simulated drift, retrain. |

---

## Tech stack

Python · pandas · scikit-learn · joblib · FastAPI · uvicorn · Streamlit · matplotlib · seaborn  

---

## Repository layout

```
├── api/                 # FastAPI app
├── dashboard/           # Streamlit app
├── data/                # cleaned_data.csv, new_data.csv (generated, not in Git)
├── model/               # .pkl, metrics.json, model_metadata.json (see .gitignore)
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── drift.py
│   ├── retrain.py
│   └── new_data_io.py   # robust CSV reader for mixed prediction logs
├── .github/workflows/   # CI
├── requirements.txt
└── .streamlit/          # theme / toolbar options for Streamlit
```

**Note:** Large artifacts (`household_power_consumption.txt`, cleaned CSV, pickles) are **gitignored** so the repo stays under GitHub size limits. CI and local runs reproduce them.

---

## Quick start

### 1. Clone and environment

```bash
git clone https://github.com/RishithaKorrapati/Energy-Consumption-Drift-Detection-Auto-Retraining-ML-Pipeline.git
cd Energy-Consumption-Drift-Detection-Auto-Retraining-ML-Pipeline
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Obtain the raw dataset

Either:

- Download from UCI and place **`household_power_consumption.txt`** in the project root, or  
- Let **CI** fetch it (see workflow), or use the same `curl` + `unzip` pattern from `.github/workflows/ci.yml`.

### 3. Full local pipeline (full ~2M rows)

```bash
python scripts/preprocess.py
python scripts/train.py
```

### 4. API

```bash
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

- Docs: `http://127.0.0.1:8000/docs`  
- Health: `GET /health`  
- Root `/` returns 404 by design.

### 5. Dashboard

```bash
streamlit run dashboard/app.py
```

Uses **`model/model.pkl`** and reads/writes the same `data/` and `model/` files as the scripts.

### 6. Drift and retrain (CLI)

```bash
python scripts/drift.py --reference data/cleaned_data.csv --current data/new_data.csv --threshold 0.35
python scripts/retrain.py --reference data/cleaned_data.csv --current data/new_data.csv --threshold 0.35
```

---

## CI/CD

On push/PR to `main` or `master`, the workflow installs dependencies, downloads the dataset when absent, runs **preprocess** (sampled rows for speed), **train**, builds a shifted `new_data` window to simulate drift, then runs **drift** and **retrain**. See `.github/workflows/ci.yml`.

---

## Windows tip (PowerShell)

Use `Invoke-RestMethod` for `POST /predict` instead of Unix-style `curl` flags. Example:

```powershell
$body = @{ hour = 18; day = 16; month = 12 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

---

## Design notes

- **Drift on API logs** uses `predicted_power` when `Global_active_power` is absent (operational proxy); production systems often compare on **realized tallies** or features as well.
- **Concurrent `/predict`** and dashboard logging use locked CSV appends to avoid corrupted rows.
- Tune **`--threshold`** on `drift.py` / `retrain.py` / the dashboard slider to control sensitivity.

---

## License / attribution

Use the UCI dataset in line with its terms. This project is provided as-is for learning and demonstration.

---

## Author

**Rishitha Korrapati** — [Energy-Consumption-Drift-Detection-Auto-Retraining-ML-Pipeline](https://github.com/RishithaKorrapati/Energy-Consumption-Drift-Detection-Auto-Retraining-ML-Pipeline)

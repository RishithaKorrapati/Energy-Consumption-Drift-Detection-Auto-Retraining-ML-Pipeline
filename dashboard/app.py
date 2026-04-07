"""
Streamlit dashboard for the energy drift-detection pipeline.

Shows predictions context, old vs new distribution plots, drift status, metrics, and model version info.
"""

from __future__ import annotations

import html as html_module
import json
import sys
import threading
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from drift import detect_drift, load_power_series, write_drift_status
from new_data_io import read_new_data_csv

CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned_data.csv"
NEW_DATA_PATH = PROJECT_ROOT / "data" / "new_data.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
_log_lock = threading.Lock()
METRICS_PATH = PROJECT_ROOT / "model" / "metrics.json"
METADATA_PATH = PROJECT_ROOT / "model" / "model_metadata.json"
DRIFT_PATH = PROJECT_ROOT / "model" / "drift_status.json"

# Chart palette (reference vs incoming)
COLOR_REF = "#5eb8c9"
COLOR_NEW = "#e8a04f"
COLOR_GRID = "#2a3142"
COLOR_TEXT = "#c8cdd8"
COLOR_BG = "#12151c"
COLOR_LEGEND_BG = "#1c2230"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _incoming_power_for_plot(df: pd.DataFrame) -> pd.Series:
    """Match drift logic: prefer measured power, else predictions."""
    if df.empty:
        return pd.Series(dtype=float)
    gap = pd.to_numeric(df["Global_active_power"], errors="coerce") if "Global_active_power" in df.columns else None
    pred = pd.to_numeric(df["predicted_power"], errors="coerce") if "predicted_power" in df.columns else None
    if gap is not None and pred is not None:
        s = gap.fillna(pred)
    elif gap is not None:
        s = gap
    else:
        s = pred if pred is not None else pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce").dropna()


def _sample_series(df: pd.DataFrame, column_name: str, max_rows: int = 50_000) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if len(s) > max_rows:
        s = s.sample(max_rows, random_state=42)
    return s


def _apply_chart_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLOR_BG,
            "axes.facecolor": COLOR_BG,
            "axes.edgecolor": COLOR_GRID,
            "axes.labelcolor": COLOR_TEXT,
            "axes.titlecolor": "#eef1f6",
            "text.color": COLOR_TEXT,
            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
            "grid.color": COLOR_GRID,
            "grid.alpha": 0.45,
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.titlesize": 13,
        }
    )


def _inject_theme() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg0: #080a0d;
                --bg1: #0e1117;
                --surface: #141922;
                --surface2: #1c2230;
                --border: rgba(255,255,255,0.08);
                --text: #e6e9ef;
                --muted: #8b93a7;
                --accent: #5eb8c9;
                --accent2: #e8a04f;
                --ok: #6bcf9b;
                --alert: #f07178;
            }
            [data-testid="stAppViewContainer"] {
                background: radial-gradient(1200px 600px at 10% -10%, rgba(94,184,201,0.08), transparent 50%),
                            radial-gradient(800px 400px at 90% 0%, rgba(232,160,79,0.06), transparent 45%),
                            linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
            }
            [data-testid="stHeader"] { background: transparent; }
            .block-container { padding-top: 2.5rem !important; padding-bottom: 3rem !important; max-width: 1120px !important; }
            h1, h2, h3 { font-family: 'Instrument Serif', Georgia, serif !important; font-weight: 400 !important; letter-spacing: -0.02em; }
            h1 { color: var(--text) !important; font-size: 2.35rem !important; line-height: 1.15 !important; margin-bottom: 0.35rem !important; }
            h2 { color: var(--text) !important; font-size: 1.45rem !important; margin-top: 1.75rem !important; margin-bottom: 0.75rem !important; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
            h3 { color: var(--muted) !important; font-size: 0.95rem !important; text-transform: uppercase; letter-spacing: 0.14em; font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; margin-top: 1.25rem !important; }
            p, span, label, [data-testid="stMarkdownContainer"] p { font-family: 'DM Sans', sans-serif !important; color: var(--muted) !important; }
            .dash-hero-sub { font-family: 'DM Sans', sans-serif; color: var(--muted); font-size: 1.05rem; margin-top: 0.25rem; margin-bottom: 1.75rem; max-width: 52ch; line-height: 1.55; }
            .metric-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 0.5rem; }
            @media (max-width: 900px) { .metric-row { grid-template-columns: 1fr; } }
            .metric-card {
                background: linear-gradient(165deg, var(--surface) 0%, var(--surface2) 100%);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1.25rem 1.35rem;
                box-shadow: 0 12px 40px rgba(0,0,0,0.35);
            }
            .metric-card .lbl {
                font-family: 'DM Sans', sans-serif;
                font-size: 0.68rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.16em;
                color: var(--muted);
                margin-bottom: 0.5rem;
            }
            .metric-card .val {
                font-family: 'DM Sans', sans-serif;
                font-size: 1.65rem;
                font-weight: 600;
                color: var(--text);
                line-height: 1.2;
            }
            .metric-card .val.small { font-size: 1.05rem; font-weight: 500; }
            .pill {
                display: inline-block;
                font-family: 'DM Sans', sans-serif;
                font-size: 0.72rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                margin-top: 0.5rem;
            }
            .pill.ok { background: rgba(107,207,155,0.15); color: var(--ok); border: 1px solid rgba(107,207,155,0.35); }
            .pill.bad { background: rgba(240,113,120,0.12); color: var(--alert); border: 1px solid rgba(240,113,120,0.35); }
            .pill.unk { background: rgba(139,147,167,0.12); color: var(--muted); border: 1px solid var(--border); }
            .perf-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; }
            .perf-box {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1rem 1.2rem;
                font-family: 'DM Sans', sans-serif;
            }
            .perf-box .p-lbl { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
            .perf-box .p-val { font-size: 1.5rem; font-weight: 600; color: var(--accent); margin-top: 0.35rem; }
            .drift-table { width: 100%; border-collapse: collapse; font-family: 'DM Sans', sans-serif; font-size: 0.88rem; }
            .drift-table td { padding: 0.45rem 0.65rem; border-bottom: 1px solid var(--border); color: var(--muted); }
            .drift-table td:first-child { color: var(--text); font-weight: 500; width: 42%; }
            .footer-note { font-family: 'DM Sans', sans-serif; font-size: 0.78rem; color: var(--muted); opacity: 0.85; margin-top: 2.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card_html(label: str, value: str, *, pill_html: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="lbl">{label}</div>
        <div class="val">{value}</div>
        {pill_html}
    </div>
    """


@st.cache_resource(show_spinner=False)
def _load_sklearn_model():
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)


def _append_new_data_row(hour: int, day: int, month: int, predicted: float) -> None:
    """Same schema as the FastAPI logger (datetime + features + predicted_power)."""
    dt = datetime(2006, month, day, hour)
    row = {
        "datetime": dt.isoformat(sep=" "),
        "hour": hour,
        "day": day,
        "month": month,
        "predicted_power": predicted,
    }
    df = pd.DataFrame([row])
    NEW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock:
        write_header = not NEW_DATA_PATH.exists()
        df.to_csv(NEW_DATA_PATH, mode="a", index=False, header=write_header)


@st.cache_data(show_spinner="Loading reference series for drift…")
def _reference_power_array(_cleaned_mtime: float) -> np.ndarray:
    """Full reference distribution (cached until cleaned_data.csv changes)."""
    return load_power_series(CLEANED_PATH).to_numpy(dtype=float)


def _emit_html(fragment: str) -> None:
    """
    Render HTML fragments as real DOM.

    Streamlit often shows structural HTML (divs, tables) as literal text in st.markdown
    even with unsafe_allow_html=True. st.html (1.31+) renders correctly.
    """
    html_fn = getattr(st, "html", None)
    if html_fn is not None:
        html_fn(fragment)
    else:
        st.markdown(fragment, unsafe_allow_html=True)


st.set_page_config(page_title="Energy drift monitor", layout="wide", initial_sidebar_state="collapsed")
_inject_theme()

meta = _load_json(METADATA_PATH)
metrics = _load_json(METRICS_PATH)
drift = _load_json(DRIFT_PATH)

_emit_html(
    """
    <h1>Energy drift monitor</h1>
    <p class="dash-hero-sub">Household power pipeline — model health, distribution shift, and retraining signals in one view.</p>
    """
)

st.markdown("### Interactive: log predictions & run drift")
st.caption(
    "Visitors can enter the same features as the API (hour, day, month). "
    "The app loads **model.pkl**, predicts kW, appends to **data/new_data.csv**, then you can compare that file to **cleaned_data.csv** and refresh drift below."
)

if "ui_predict_msg" in st.session_state:
    st.success(st.session_state.pop("ui_predict_msg"))
if "ui_drift_msg" in st.session_state:
    kind, text = st.session_state.pop("ui_drift_msg")
    if kind == "error":
        st.error(text)
    elif kind == "warn":
        st.warning(text)
    else:
        st.success(text)

_model = _load_sklearn_model()
if _model is None:
    st.warning("No trained model found. Run `python scripts/train.py` so **model/model.pkl** exists.")
else:
    with st.form("browser_predict"):
        fc1, fc2, fc3 = st.columns(3)
        in_hour = fc1.number_input("Hour", min_value=0, max_value=23, value=18, help="0–23, local clock for the synthetic datetime row.")
        in_day = fc2.number_input("Day", min_value=1, max_value=31, value=16)
        in_month = fc3.number_input("Month", min_value=1, max_value=12, value=12)
        in_repeat = st.number_input(
            "Repeat this row (times to append)",
            min_value=1,
            max_value=5000,
            value=1,
            help="Append the same prediction N times to grow **new_data** quickly for demos.",
        )
        submitted_predict = st.form_submit_button("Predict & append to new data")

    if submitted_predict:
        x = np.array([[in_hour, in_day, in_month]], dtype=float)
        pred_val = float(_model.predict(x)[0])
        for _ in range(int(in_repeat)):
            _append_new_data_row(in_hour, in_day, in_month, pred_val)
        st.session_state["ui_predict_msg"] = (
            f"Appended **{int(in_repeat)}** row(s) to `data/new_data.csv`. "
            f"Predicted power: **{pred_val:.4f} kW**."
        )
        st.rerun()

    dcol1, dcol2 = st.columns([2, 1])
    with dcol1:
        drift_threshold_ui = st.slider(
            "Drift threshold (|Δmean| / σ_ref)",
            min_value=0.05,
            max_value=1.5,
            value=0.35,
            step=0.05,
            help="Same meaning as `python scripts/drift.py --threshold`. Above → drift detected.",
        )
    with dcol2:
        st.write("")  # vertical align
        st.write("")
        run_drift = st.button("Run drift check", type="primary", use_container_width=True)

    if run_drift:
        if not CLEANED_PATH.is_file():
            st.session_state["ui_drift_msg"] = ("error", "Missing **data/cleaned_data.csv**. Run `python scripts/preprocess.py` first.")
            st.rerun()
        elif not NEW_DATA_PATH.is_file():
            st.session_state["ui_drift_msg"] = ("warn", "No **data/new_data.csv** yet. Use the form above to append predictions.")
            st.rerun()
        else:
            try:
                ref_mtime = CLEANED_PATH.stat().st_mtime
                ref_arr = _reference_power_array(ref_mtime)
                cur_arr = load_power_series(NEW_DATA_PATH).to_numpy(dtype=float)
            except (FileNotFoundError, ValueError) as e:
                st.session_state["ui_drift_msg"] = ("error", str(e))
                st.rerun()
            else:
                if len(cur_arr) == 0:
                    st.session_state["ui_drift_msg"] = ("warn", "**new_data.csv** has no usable power / prediction values.")
                    st.rerun()
                else:
                    drift_detected, details = detect_drift(
                        ref_arr, cur_arr, threshold=float(drift_threshold_ui)
                    )
                    write_drift_status(details)
                    if drift_detected:
                        st.session_state["ui_drift_msg"] = (
                            "error",
                            "**Drift detected** — shift exceeds threshold. Consider `python scripts/retrain.py` with the same threshold.",
                        )
                    else:
                        st.session_state["ui_drift_msg"] = (
                            "ok",
                            "**No drift** — incoming window is within threshold vs full reference data.",
                        )
                    st.rerun()

drift_flag = drift.get("drift_detected")
if drift_flag is True:
    pill = '<span class="pill bad">Drift detected</span>'
elif drift_flag is False:
    pill = '<span class="pill ok">Stable</span>'
else:
    pill = '<span class="pill unk">Unknown</span>'

version = meta.get("model_version", "—")
last_train = meta.get("last_trained_at") or meta.get("last_retrained_at") or "—"
if isinstance(last_train, str) and len(last_train) > 19:
    last_train = last_train[:19].replace("T", " ") + " UTC"

row_html = f"""
<div class="metric-row">
  {_metric_card_html("Model version", html_module.escape(str(version)))}
  {_metric_card_html("Last trained / retrained", f'<span class="val small">{html_module.escape(last_train)}</span>')}
  {_metric_card_html("Drift detected", html_module.escape("Yes" if drift_flag is True else "No" if drift_flag is False else "—"), pill_html=pill)}
</div>
"""
_emit_html(row_html)

st.markdown("### Model performance")
if metrics:
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")
    mae_s = f"{mae:.4f}" if mae is not None else "—"
    rmse_s = f"{rmse:.4f}" if rmse is not None else "—"
    _emit_html(
        f"""
        <div class="perf-grid">
            <div class="perf-box"><div class="p-lbl">MAE (hold-out)</div><div class="p-val">{html_module.escape(mae_s)}</div></div>
            <div class="perf-box"><div class="p-lbl">RMSE (hold-out)</div><div class="p-val">{html_module.escape(rmse_s)}</div></div>
        </div>
        """
    )
else:
    st.info("Run `scripts/train.py` to produce metrics.")

st.markdown("### Drift diagnostics")
if drift:
    rows = []
    for k, v in drift.items():
        if k == "drift_detected":
            continue
        if isinstance(v, float):
            v = f"{v:.6g}" if abs(v) < 1e5 else f"{v:.4g}"
        rows.append(
            f"<tr><td>{html_module.escape(str(k))}</td><td>{html_module.escape(str(v))}</td></tr>"
        )
    _emit_html('<table class="drift-table">' + "".join(rows) + "</table>")
else:
    st.caption("Run `scripts/drift.py` or `scripts/retrain.py` to refresh drift status.")

st.markdown("### Distributions")
if not CLEANED_PATH.is_file():
    st.warning("Cleaned data not found. Run preprocessing and training first.")
else:
    cleaned = pd.read_csv(CLEANED_PATH, parse_dates=["datetime"], nrows=200_000)
    ref_series = _sample_series(cleaned, "Global_active_power")

    new_df = read_new_data_csv(NEW_DATA_PATH) if NEW_DATA_PATH.is_file() else pd.DataFrame()
    raw_incoming = _incoming_power_for_plot(new_df)
    new_series = raw_incoming if len(raw_incoming) <= 50_000 else raw_incoming.sample(50_000, random_state=42)
    new_label = "Incoming (measured or predicted)"

    _apply_chart_style()
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": COLOR_BG, "figure.facecolor": COLOR_BG})

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=120)
    ax.set_facecolor(COLOR_BG)
    if len(ref_series) > 0:
        sns.histplot(ref_series, color=COLOR_REF, label="Reference (cleaned)", kde=True, ax=ax, stat="density", linewidth=0)
    if len(new_series) > 0:
        sns.histplot(new_series, color=COLOR_NEW, label=new_label, kde=True, ax=ax, stat="density", linewidth=0)
    ax.set_xlabel("Active power / prediction (kW)", color=COLOR_TEXT)
    ax.set_title("Reference vs incoming distribution", color="#eef1f6", pad=12, fontsize=12, fontweight="500")
    leg = ax.legend(frameon=True, facecolor=COLOR_LEGEND_BG, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT)
    for text in leg.get_texts():
        text.set_color(COLOR_TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)

    if CLEANED_PATH.is_file() and len(new_df) > 0 and "datetime" in cleaned.columns and "datetime" in new_df.columns:
        st.markdown("#### Recent timeline (sampled)")
        t_ref = cleaned.sort_values("datetime").tail(2_000)
        t_new = new_df.copy()
        if "datetime" in t_new.columns:
            t_new["datetime"] = pd.to_datetime(t_new["datetime"], errors="coerce")
            t_new = t_new.dropna(subset=["datetime"]).sort_values("datetime").tail(2_000)
            y_ref = pd.to_numeric(t_ref["Global_active_power"], errors="coerce")
            g2 = (
                pd.to_numeric(t_new["Global_active_power"], errors="coerce")
                if "Global_active_power" in t_new.columns
                else None
            )
            p2 = (
                pd.to_numeric(t_new["predicted_power"], errors="coerce")
                if "predicted_power" in t_new.columns
                else None
            )
            if g2 is not None and p2 is not None:
                y_new = g2.fillna(p2)
            elif g2 is not None:
                y_new = g2
            elif p2 is not None:
                y_new = p2
            else:
                y_new = None
            if y_new is not None:
                fig2, ax2 = plt.subplots(figsize=(11, 3.4), dpi=120)
                ax2.set_facecolor(COLOR_BG)
                ax2.plot(t_ref["datetime"], y_ref, label="Reference tail", color=COLOR_REF, alpha=0.85, linewidth=1.1)
                ax2.plot(t_new["datetime"], y_new, label="Incoming tail", color=COLOR_NEW, alpha=0.85, linewidth=1.1)
                ax2.set_ylabel("kW", color=COLOR_TEXT)
                ax2.legend(frameon=True, facecolor=COLOR_LEGEND_BG, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT)
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)
                st.pyplot(fig2, bbox_inches="tight")
                plt.close(fig2)

_emit_html(
    f'<p class="footer-note">Project root · <code style="color:#5eb8c9;">{html_module.escape(str(PROJECT_ROOT))}</code></p>'
)

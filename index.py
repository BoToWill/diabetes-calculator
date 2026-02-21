"""
GlyPro v5.0 — Повний щоденник діабетика 1 типу
⚠️  Тільки для ознайомлення. Консультуйтеся з ендокринологом.
"""

import streamlit as st
import pandas as pd
import json
import os
import math
from datetime import datetime, timedelta
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

APP_VERSION = "5.0"
DATA_FILE   = "diabetes_data.json"
BACKUP_FILE = "diabetes_backup.json"

GLUCOSE_ZONES = [
    (0,    3.9,  "🔴 Гіпоглікемія",   "danger",   "#f87171", "НЕБЕЗПЕКА! Терміново з'їжте 15 г швидких вуглеводів!"),
    (4.0,  4.4,  "🟠 Низький",         "low",      "#fb923c", "Рекомендується невеликий перекус."),
    (4.5,  7.8,  "🟢 Цільовий",        "target",   "#34d399", "Відмінно! Глюкоза в нормі ✨"),
    (7.9,  10.0, "🟡 Підвищений",      "elevated", "#fbbf24", "Розгляньте невелику корекційну дозу."),
    (10.1, 13.9, "🟠 Високий",          "high",     "#fb923c", "Потрібна корекція. Перевірте кетони."),
    (14.0, 99.0, "🔴 Дуже високий",     "danger",   "#f87171", "НЕБЕЗПЕКА! Введіть інсулін, перевірте кетони!"),
]

# Time-of-day ISF multipliers (ранок має вищу резистентність)
TOD_ISF_FACTORS = {
    "🌅 Ранок (6–10)":   0.85,
    "☀️ День (10–17)":   1.00,
    "🌆 Вечір (17–21)":  1.05,
    "🌙 Ніч (21–6)":     1.10,
}

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9ca3af", family="'DM Mono', monospace", size=11),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)",
               showline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)",
               showline=False),
    margin=dict(l=40, r=20, t=44, b=36),
    hoverlabel=dict(bgcolor="#181c23", bordercolor="rgba(255,255,255,0.1)",
                    font=dict(color="#eef0f5", family="'Outfit', sans-serif")),
)


# ══════════════════════════════════════════════════════════════════════════════
#  PURE MATH / MEDICAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_zone(g: float) -> tuple:
    for lo, hi, label, key, color, msg in GLUCOSE_ZONES:
        if lo <= g <= hi:
            return label, key, color, msg
    return "❓", "unknown", "#6b7280", ""


def calc_bu(carbs100: float, weight_g: float, bu_w: float) -> dict:
    carbs = (carbs100 * weight_g) / 100
    return {
        "carbs":    round(carbs, 1),
        "bu":       round(carbs / bu_w, 2),
        "calories": round(carbs * 4, 0),
        "gl":       round(carbs * 50 / 100, 1),
    }


def iob_remaining(units: float, minutes_ago: float,
                  duration_h: float = 4.0, peak_h: float = 1.0) -> float:
    """Bi-exponential IOB decay (simplified Bergman model)."""
    if minutes_ago <= 0 or units <= 0:
        return units
    t   = minutes_ago / 60
    dur = duration_h
    if t >= dur:
        return 0.0
    # Cubic polynomial approximation
    pct = 1 - (t / dur) ** 2 * (3 - 2 * t / dur)
    return max(0.0, round(units * pct, 2))


def estimate_hba1c(avg_glucose_mmol: float) -> float:
    """Nathan formula: HbA1c (%) = (avg_glucose_mg + 46.7) / 28.7"""
    avg_mg = avg_glucose_mmol * 18.0
    return round((avg_mg + 46.7) / 28.7, 1)


def glucose_prediction_curve(current_g: float, dose_units: float,
                              carbs_g: float, isf: float, cr: float,
                              minutes: int = 240) -> tuple:
    """Simple glucose prediction over time after bolus."""
    times, values = [], []
    for t in range(0, minutes + 1, 10):
        # Carb absorption (triangular peak at 45 min)
        carb_peak = 45
        if t <= carb_peak:
            absorbed_pct = t / carb_peak
        else:
            absorbed_pct = max(0, 1 - (t - carb_peak) / 135)
        glucose_from_carbs = (carbs_g * absorbed_pct * 0.55) / 18  # mmol/L rough

        # Insulin action (peak at 60 min)
        ins_peak = 60
        if dose_units > 0 and t > 0:
            ins_action = dose_units * isf * math.exp(-((t - ins_peak) ** 2) / (2 * 50 ** 2))
            ins_action = min(ins_action, dose_units * isf * 0.6)
        else:
            ins_action = 0

        g = current_g + glucose_from_carbs - ins_action * (t / (ins_peak * 2))
        times.append(t)
        values.append(round(max(2.5, g), 1))
    return times, values


def get_meal_type(hour: int) -> str:
    if  5 <= hour < 11: return "🌅 Сніданок"
    if 11 <= hour < 15: return "☀️ Обід"
    if 15 <= hour < 18: return "🍵 Перекус"
    return "🌙 Вечеря"


def tir_stats(logs: list) -> dict:
    if not logs:
        return {"hypo": 0, "low": 0, "target": 0, "high": 0, "very_high": 0, "avg": 0, "n": 0}
    levels = [e["level"] for e in logs]
    n = len(levels)
    return {
        "hypo":      round(sum(1 for g in levels if g < 4.0)  / n * 100),
        "low":       round(sum(1 for g in levels if 4.0 <= g < 4.5) / n * 100),
        "target":    round(sum(1 for g in levels if 4.5 <= g <= 7.8) / n * 100),
        "high":      round(sum(1 for g in levels if 7.8 < g <= 13.9) / n * 100),
        "very_high": round(sum(1 for g in levels if g > 13.9) / n * 100),
        "avg":       round(np.mean(levels), 1),
        "std":       round(np.std(levels), 1),
        "n":         n,
        "hba1c":     estimate_hba1c(np.mean(levels)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> dict:
    for path in [DATA_FILE, BACKUP_FILE]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if "meal_data" in d:
                    return d
            except Exception:
                continue
    return {}


def save_data() -> None:
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as s, open(BACKUP_FILE, "w") as d:
                d.write(s.read())
        ss = st.session_state
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "meal_data":       ss.meal_data,
                "bu_weight":       ss.bu_weight,
                "daily_totals":    ss.daily_totals,
                "product_history": ss.product_history,
                "product_freq":    ss.product_freq,
                "user_profile":    ss.user_profile,
                "glucose_logs":    ss.glucose_logs,
                "meal_patterns":   ss.meal_patterns,
                "meal_templates":  ss.meal_templates,
                "insulin_profile": ss.insulin_profile,
                "last_saved":      datetime.now().isoformat(),
                "version":         APP_VERSION,
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"❌ Збереження: {e}")


@st.cache_data(show_spinner=False)
def load_product_db() -> dict:
    db: dict = {}
    try:
        with open("table.csv", "r", encoding="utf-8") as f:
            for line in f.readlines()[2:]:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 5:
                    db.setdefault(parts[0], []).append({
                        "name": parts[1], "calories": float(parts[2]),
                        "protein": float(parts[3]), "carbs": float(parts[4]),
                    })
    except Exception as e:
        st.warning(f"База продуктів: {e}")
    return db


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "meal_data":       [],
        "bu_weight":       12,
        "daily_totals":    {},
        "product_history": [],
        "product_freq":    {},
        "user_profile": {
            "name": "", "age": 25, "weight": 60, "height": 165,
            "activity": "medium", "insulin_type": "rapid",
            "target_min": 4.5, "target_max": 7.8,
            "tdd": 35.0,
        },
        "glucose_logs":    [],
        "meal_patterns":   {},
        "meal_templates":  {},
        "insulin_profile": {
            "cr": 10.0, "isf": 2.5, "iob_duration": 4.0,
            "active_doses": [],
        },
        "_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state._loaded:
        d = load_data()
        if d:
            for key in ["meal_data","bu_weight","daily_totals","product_history",
                        "product_freq","glucose_logs","meal_patterns","meal_templates"]:
                if key in d:
                    st.session_state[key] = d[key]
            if "user_profile"    in d: st.session_state.user_profile.update(d["user_profile"])
            if "insulin_profile" in d: st.session_state.insulin_profile.update(d["insulin_profile"])
        st.session_state._loaded = True


# ══════════════════════════════════════════════════════════════════════════════
#  MEAL LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def get_totals() -> dict:
    data = st.session_state.meal_data
    if not data:
        return {"carbs": 0, "bu": 0, "cal": 0, "gl": 0}
    return {
        "carbs": round(sum(i["Вугл."] for i in data), 1),
        "bu":    round(sum(i["ХО"]    for i in data), 2),
        "cal":   round(sum(i.get("Ккал", 0) for i in data), 0),
        "gl":    round(sum(i.get("ГН", 0)   for i in data), 1),
    }


def add_product(name: str, carbs100: float, weight: float,
                protein: float = 0, calories_per100: float = 0) -> bool:
    name = name.strip()
    if not name or len(name) < 2:
        st.error("Введіть назву продукту (мін. 2 символи)"); return False
    if not (0 <= carbs100 <= 100):
        st.error("Вуглеводи: 0–100 г на 100 г"); return False
    if not (1 <= weight <= 5000):
        st.error("Вага: 1–5000 г"); return False

    bu_w = st.session_state.bu_weight
    c    = calc_bu(carbs100, weight, bu_w)

    # if real calories known from DB
    real_kcal = round(calories_per100 * weight / 100, 0) if calories_per100 else c["calories"]

    st.session_state.meal_data.append({
        "Продукт": name,
        "Вага":    int(weight),
        "Вугл.":   c["carbs"],
        "ХО":      c["bu"],
        "Ккал":    real_kcal,
        "ГН":      c["gl"],
        "Час":     datetime.now().strftime("%H:%M"),
        "Дата":    datetime.now().strftime("%Y-%m-%d"),
    })

    freq = st.session_state.product_freq
    freq[name] = freq.get(name, 0) + 1
    if name not in st.session_state.product_history:
        st.session_state.product_history.append(name)

    meal_t = get_meal_type(datetime.now().hour)
    st.session_state.meal_patterns.setdefault(meal_t, []).append(
        {"product": name, "bu": c["bu"], "carbs": c["carbs"]}
    )
    save_data()
    return True


def save_meal_snapshot():
    if not st.session_state.meal_data:
        st.warning("Немає даних для збереження"); return
    today = datetime.now().strftime("%Y-%m-%d")
    st.session_state.daily_totals.setdefault(today, []).append({
        "timestamp": datetime.now().isoformat(),
        "data":      st.session_state.meal_data.copy(),
        "totals":    get_totals(),
        "meal_type": get_meal_type(datetime.now().hour),
    })
    save_data()
    st.success("💾 Прийом збережено в журнал!")


# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"], .stApp { font-family: 'Outfit', sans-serif !important; }
.stApp { background: #080a0e !important; color: #eef0f5 !important; }
.block-container { padding: 1.5rem 2rem 4rem !important; max-width: 1400px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0f15 0%, #0a0c11 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stSidebar"] * { color: #d1d5db !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] strong { color: #eef0f5 !important; }
section[data-testid="stSidebar"] > div { padding: 1.5rem 1rem !important; }

/* ── Inputs ── */
.stTextInput input,.stNumberInput input,
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div:first-child {
    background: #13161e !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    color: #eef0f5 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput input:focus,.stNumberInput input:focus {
    border-color: rgba(232,80,106,.5) !important;
    box-shadow: 0 0 0 3px rgba(232,80,106,.07) !important;
}
.stSelectbox label,.stNumberInput label,.stTextInput label {
    font-size: 11px !important; color: #6b7280 !important;
    font-weight: 600 !important; letter-spacing: .06em !important;
    text-transform: uppercase !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg,#e8506a,#f97b4f) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-family: 'Outfit',sans-serif !important;
    font-weight: 600 !important; font-size: 13px !important;
    padding: 10px 20px !important; letter-spacing: .02em !important;
    box-shadow: 0 4px 16px rgba(232,80,106,.25) !important;
    transition: all .2s !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 24px rgba(232,80,106,.4) !important; }
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button[kind="secondary"] {
    background: #13161e !important; border: 1px solid rgba(255,255,255,.07) !important;
    box-shadow: none !important; color: #9ca3af !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: rgba(232,80,106,.3) !important; color: #eef0f5 !important;
    transform: none !important; box-shadow: none !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important; gap: 2px !important;
    border-bottom: 1px solid rgba(255,255,255,.06) !important; padding-bottom: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #4b5563 !important;
    font-family: 'Outfit',sans-serif !important; font-size: 13px !important;
    font-weight: 600 !important; padding: 10px 18px !important;
    border-radius: 8px 8px 0 0 !important; border: none !important;
    letter-spacing: .02em !important; transition: color .15s !important;
}
.stTabs [aria-selected="true"] {
    color: #eef0f5 !important; background: transparent !important;
    border-bottom: 2px solid #e8506a !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) { color: #9ca3af !important; }
.stTabs [data-baseweb="tab-panel"] { background: transparent !important; padding-top: 24px !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #111318 !important; border: 1px solid rgba(255,255,255,.06) !important;
    border-radius: 14px !important; padding: 16px 20px !important;
}
[data-testid="stMetricValue"] { font-family: 'DM Mono',monospace !important; font-size: 1.4rem !important; }
[data-testid="stMetricLabel"] { font-size: 11px !important; color: #6b7280 !important; text-transform: uppercase !important; letter-spacing:.06em !important; }
[data-testid="stMetricDelta"] { font-family: 'DM Mono',monospace !important; font-size: 12px !important; }

/* ── Expander ── */
details > summary {
    background: #111318 !important; border: 1px solid rgba(255,255,255,.06) !important;
    border-radius: 10px !important; padding: 12px 16px !important;
    font-family: 'Outfit',sans-serif !important; font-size: 13px !important;
    font-weight: 600 !important; color: #d1d5db !important;
    list-style: none !important; cursor: pointer !important;
    transition: border-color .2s !important;
}
details > summary:hover { border-color: rgba(232,80,106,.3) !important; }
details[open] > summary { border-radius: 10px 10px 0 0 !important; border-bottom-color: transparent !important; }
details > div {
    background: #111318 !important; border: 1px solid rgba(255,255,255,.06) !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important; padding: 12px 16px !important;
}

/* ── DataFrame ── */
.stDataFrame > div { border-radius: 14px !important; overflow: hidden !important; }
.stDataFrame [data-testid="stDataFrameResizable"] { background: #111318 !important; }

/* ── Alerts ── */
.stAlert { border-radius: 10px !important; font-family: 'Outfit',sans-serif !important; font-size: 13px !important; }

/* ── Progress ── */
.stProgress > div > div { background: linear-gradient(90deg,#e8506a,#f97b4f) !important; border-radius: 9px !important; }
.stProgress > div { background: #181c23 !important; border-radius: 9px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.08); border-radius: 5px; }

/* ── Custom HTML Components ── */
.glyco-title {
    font-family: 'DM Serif Display',serif;
    background: linear-gradient(135deg,#e8506a,#f97b4f);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    line-height: 1.1; letter-spacing: -1px;
}
.card {
    background: #111318; border: 1px solid rgba(255,255,255,.06);
    border-radius: 16px; padding: 20px 24px; margin-bottom: 16px;
}
.card-sm { padding: 14px 16px; border-radius: 12px; }
.card-accent {
    background: linear-gradient(135deg,rgba(232,80,106,.07),rgba(249,123,79,.04));
    border-color: rgba(232,80,106,.18);
}
.result-hero { text-align: center; padding: 28px 24px; }
.result-num {
    font-family: 'DM Serif Display',serif; font-size: 4rem; line-height: 1;
    background: linear-gradient(135deg,#e8506a,#f97b4f);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    display: block;
}
.result-unit { font-size: 13px; color: #6b7280; margin-top: 4px; display: block; }
.kpi-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin: 16px 0; }
.kpi { background: rgba(255,255,255,.03); border-radius: 10px; padding: 12px 14px; text-align: center; }
.kpi-v { font-family: 'DM Mono',monospace; font-size: 1.15rem; font-weight: 500; display: block; }
.kpi-l { font-size: 10px; color: #6b7280; text-transform: uppercase; letter-spacing: .06em; margin-top: 2px; display: block; }
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 14px; border-radius: 100px; font-size: 12px; font-weight: 600;
    border: 1px solid; margin: 2px;
}
.badge-target   { background:rgba(52,211,153,.1);  color:#34d399; border-color:rgba(52,211,153,.25); }
.badge-low      { background:rgba(251,146,60,.1);  color:#fb923c; border-color:rgba(251,146,60,.25); }
.badge-danger   { background:rgba(248,113,113,.1); color:#f87171; border-color:rgba(248,113,113,.25); }
.badge-elevated { background:rgba(251,191,36,.1);  color:#fbbf24; border-color:rgba(251,191,36,.25); }
.badge-info     { background:rgba(96,165,250,.1);  color:#60a5fa; border-color:rgba(96,165,250,.25); }
.badge-high     { background:rgba(251,146,60,.1);  color:#fb923c; border-color:rgba(251,146,60,.25); }
.divider { border: none; border-top: 1px solid rgba(255,255,255,.05); margin: 20px 0; }
.section-title { font-size: 11px; color: #4b5563; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; margin-bottom: 12px; }
.warn-bar {
    display: flex; align-items: flex-start; gap: 8px;
    background: rgba(251,191,36,.06); border: 1px solid rgba(251,191,36,.2);
    border-radius: 10px; padding: 10px 14px; font-size: 12px; color: #fbbf24; line-height: 1.5;
}
.tag {
    display: inline-block; background: #181c23; border: 1px solid rgba(255,255,255,.07);
    border-radius: 6px; padding: 3px 10px; font-size: 12px; color: #9ca3af; margin: 2px;
    cursor: pointer; transition: all .15s;
}
.iob-bar { background: #181c23; border-radius: 10px; padding: 12px 14px; }
.tir-legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
.tir-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #9ca3af; }
.tir-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def glucose_ring_html(value: Optional[float]) -> str:
    if value is None:
        return """
        <div style="display:flex;flex-direction:column;align-items:center;padding:16px 0 8px">
          <svg width="150" height="150" viewBox="0 0 150 150">
            <circle cx="75" cy="75" r="62" fill="none" stroke="#1a1d24" stroke-width="10"/>
          </svg>
          <div style="font-size:12px;color:#4b5563;margin-top:-70px;padding-bottom:50px">введіть значення</div>
        </div>"""
    label, key, color, msg = get_zone(value)
    pct  = min(1.0, max(0.0, (value - 2.0) / 18.0))
    circ = 2 * math.pi * 62
    dash = circ * pct
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;padding:12px 0 4px">
      <div style="position:relative;width:150px;height:150px">
        <svg width="150" height="150" viewBox="0 0 150 150" style="transform:rotate(-90deg)">
          <circle cx="75" cy="75" r="62" fill="none" stroke="#1a1d24" stroke-width="10"/>
          <circle cx="75" cy="75" r="62" fill="none" stroke="{color}" stroke-width="10"
            stroke-linecap="round" stroke-dasharray="{dash:.1f} {circ:.1f}"
            style="transition:all 0.8s cubic-bezier(.4,0,.2,1)"/>
        </svg>
        <div style="position:absolute;inset:0;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;gap:1px">
          <span style="font-family:'DM Serif Display',serif;font-size:2.1rem;
                        color:{color};line-height:1">{value}</span>
          <span style="font-size:10px;color:#6b7280">ммоль/л</span>
        </div>
      </div>
      <div style="margin-top:8px">
        <span class="badge badge-{key}">{label}</span>
      </div>
      <div style="font-size:11px;color:#6b7280;text-align:center;max-width:160px;
                  line-height:1.4;margin-top:6px;min-height:30px">{msg}</div>
    </div>"""


def mini_card(title: str, value: str, color: str = "#e8506a", sub: str = "") -> str:
    return f"""
    <div class="card card-sm" style="text-align:center">
      <div style="font-size:10px;color:#4b5563;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:6px">{title}</div>
      <div style="font-family:'DM Mono',monospace;font-size:1.4rem;
                  font-weight:500;color:{color}">{value}</div>
      {f'<div style="font-size:11px;color:#6b7280;margin-top:3px">{sub}</div>' if sub else ''}
    </div>"""


def result_card(bu: float, carbs: float, cal: float, gl: float) -> str:
    return f"""
    <div class="card card-accent result-hero">
      <span style="font-size:11px;color:#6b7280;text-transform:uppercase;
                   letter-spacing:.1em;margin-bottom:8px;display:block">Разом за прийом</span>
      <span class="result-num">{bu}</span>
      <span class="result-unit">хлібних одиниць</span>
      <div class="kpi-row" style="margin-top:20px">
        <div class="kpi">
          <span class="kpi-v" style="color:#34d399">{carbs} г</span>
          <span class="kpi-l">Вуглеводи</span>
        </div>
        <div class="kpi">
          <span class="kpi-v" style="color:#60a5fa">{int(cal)} ккал</span>
          <span class="kpi-l">Калорії</span>
        </div>
        <div class="kpi">
          <span class="kpi-v" style="color:#fbbf24">{gl}</span>
          <span class="kpi-l">Глік. навант.</span>
        </div>
      </div>
    </div>"""


def dose_card(total: float, rounded: float, meal_d: float, corr_d: float,
              iob: float, carbs: float, tod_factor: float) -> str:
    corr_sign  = "+" if corr_d >= 0 else ""
    corr_color = "#e8506a" if corr_d >= 0 else "#60a5fa"
    return f"""
    <div class="card card-accent" style="padding:28px 24px">
      <div style="text-align:center;margin-bottom:20px">
        <div style="font-size:11px;color:#6b7280;text-transform:uppercase;
                    letter-spacing:.1em;margin-bottom:6px">Рекомендована доза</div>
        <span class="result-num">{total:.2f}</span>
        <span class="result-unit">одиниць інсуліну (ОД)</span>
        <div style="margin-top:10px">
          <span style="font-family:'DM Mono',monospace;font-size:1.7rem;color:#fbbf24">{rounded:.1f} ОД</span>
          <span style="font-size:11px;color:#6b7280;margin-left:6px">округлено до 0.5</span>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
        <div class="kpi" style="text-align:left">
          <span class="kpi-l">🍽️ На їжу ({carbs} г)</span>
          <span class="kpi-v" style="color:#e8506a;margin-top:4px">{meal_d:.2f} ОД</span>
        </div>
        <div class="kpi" style="text-align:left">
          <span class="kpi-l">🔧 Корекція</span>
          <span class="kpi-v" style="color:{corr_color};margin-top:4px">{corr_sign}{corr_d:.2f} ОД</span>
        </div>
        <div class="kpi" style="text-align:left">
          <span class="kpi-l">⏳ Мінус IOB</span>
          <span class="kpi-v" style="color:#60a5fa;margin-top:4px">−{iob:.1f} ОД</span>
        </div>
        <div class="kpi" style="text-align:left">
          <span class="kpi-l">🕐 Коефіцієнт ЧД</span>
          <span class="kpi-v" style="color:#9ca3af;margin-top:4px">×{tod_factor:.2f}</span>
        </div>
      </div>
    </div>"""


def hba1c_card(hba1c: float, avg_g: float, tir_pct: float) -> str:
    hba1c_color = "#34d399" if hba1c <= 7.0 else ("#fbbf24" if hba1c <= 8.5 else "#f87171")
    hba1c_label = "Відмінно" if hba1c <= 7.0 else ("Добре" if hba1c <= 8.0 else "Потребує уваги")
    return f"""
    <div class="card" style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;padding:20px 24px">
      <div style="text-align:center">
        <div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">HbA1c (розрахунк.)</div>
        <div style="font-family:'DM Serif Display',serif;font-size:2.2rem;color:{hba1c_color};line-height:1">{hba1c}%</div>
        <div style="font-size:11px;color:{hba1c_color};margin-top:4px">{hba1c_label}</div>
      </div>
      <div style="text-align:center;border-left:1px solid rgba(255,255,255,.05);border-right:1px solid rgba(255,255,255,.05)">
        <div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Середня глюкоза</div>
        <div style="font-family:'DM Mono',monospace;font-size:2rem;color:#60a5fa;line-height:1">{avg_g}</div>
        <div style="font-size:11px;color:#6b7280;margin-top:4px">ммоль/л</div>
      </div>
      <div style="text-align:center">
        <div style="font-size:10px;color:#4b5563;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Час у нормі (TIR)</div>
        <div style="font-family:'DM Mono',monospace;font-size:2rem;color:#34d399;line-height:1">{tir_pct}%</div>
        <div style="font-size:11px;color:#6b7280;margin-top:4px">ціль: ≥ 70%</div>
      </div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="glyco-title" style="font-size:1.7rem;margin-bottom:2px">GlyPro</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:#4b5563;letter-spacing:.08em;margin-bottom:0">ЩОДЕННИК ДІАБЕТИКА</div>', unsafe_allow_html=True)
        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

        # ── Live Glucose Ring ──
        st.markdown('<div class="section-title">🩸 Поточна глюкоза</div>', unsafe_allow_html=True)
        g_now = st.number_input("Глюкоза (ммоль/л)", 0.5, 35.0, step=0.1,
                                 key="sidebar_glucose", label_visibility="collapsed")
        if g_now and g_now > 0:
            st.markdown(glucose_ring_html(g_now), unsafe_allow_html=True)
        else:
            st.markdown(glucose_ring_html(None), unsafe_allow_html=True)

        # Glucose time context
        g_time = st.selectbox("Момент вимірювання",
            ["Перед їжею","Після їжі (1 год)","Після їжі (2 год)","Вранці натще","Перед сном"],
            key="sidebar_g_time", label_visibility="collapsed")

        if st.button("📝 Записати глюкозу", use_container_width=True, key="sb_log_g"):
            if g_now > 0:
                st.session_state.glucose_logs.append({
                    "level": g_now, "time": g_time,
                    "timestamp": datetime.now().isoformat(),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                })
                save_data()
                st.success("✅ Записано!")

        # Last readings
        if st.session_state.glucose_logs:
            recent = st.session_state.glucose_logs[-4:]
            st.markdown('<div class="section-title" style="margin-top:12px">Останні вимірювання</div>', unsafe_allow_html=True)
            for e in reversed(recent):
                lv = e["level"]
                _, key, color, _ = get_zone(lv)
                badge_cls = f"badge-{key}" if key != "unknown" else "badge-info"
                ts = e["timestamp"][11:16] if "T" in e.get("timestamp","") else e.get("time","")
                st.markdown(
                    f'<span class="badge {badge_cls}" style="font-size:11px">{lv} мм</span>'
                    f'<span style="font-size:10px;color:#4b5563;margin-left:6px">{ts} · {e.get("time","")}</span>',
                    unsafe_allow_html=True)

        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

        # ── BU Weight ──
        st.markdown('<div class="section-title">⚙️ Хлібні одиниці</div>', unsafe_allow_html=True)
        bu = st.number_input("Вуглеводів в 1 ХО (г)", 8, 15,
                              st.session_state.bu_weight, step=1,
                              key="sb_bu", label_visibility="collapsed")
        if bu != st.session_state.bu_weight:
            st.session_state.bu_weight = bu
            save_data()
            st.rerun()
        st.caption(f"1 ХО = {bu} г · стандарт: 10–12 г")

        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

        # ── Quick Meal Actions ──
        if st.session_state.meal_data:
            t = get_totals()
            st.markdown(f"""
            <div class="card card-sm" style="margin-bottom:12px">
              <div class="section-title" style="margin-bottom:8px">📊 Поточний прийом</div>
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                  <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;
                       background:linear-gradient(135deg,#e8506a,#f97b4f);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                       background-clip:text;line-height:1">{t["bu"]} ХО</div>
                  <div style="font-size:11px;color:#6b7280">{t["carbs"]} г · {int(t["cal"])} ккал</div>
                </div>
                <div style="font-size:1.5rem">🍽️</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Зберегти", use_container_width=True, key="sb_save"):
                save_meal_snapshot()
        with c2:
            if st.button("🗑️ Очистити", use_container_width=True, key="sb_clear"):
                st.session_state.meal_data = []
                save_data()
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — MEAL TRACKER
# ══════════════════════════════════════════════════════════════════════════════

def tab_meal():
    # ── Manual input ──
    st.markdown('<div class="section-title">✏️ Додати вручну</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([2.5, 1.6, 1.6, 1])
    with c1:
        p_name = st.text_input("Продукт", placeholder="Гречка варена", key="m_name",
                                label_visibility="collapsed")
    with c2:
        p_c100 = st.number_input("Вугл./100г", 0.0, 100.0, step=0.5,
                                  format="%.1f", key="m_carbs", label_visibility="collapsed")
    with c3:
        p_wt   = st.number_input("Вага (г)", 1, 5000, 100, step=5,
                                  key="m_weight", label_visibility="collapsed")
    with c4:
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("➕ Додати", key="btn_add_manual", use_container_width=True):
            if add_product(p_name, p_c100, p_wt):
                st.rerun()

    # ── Frequent / Recent suggestions ──
    freq = st.session_state.product_freq
    if freq:
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8]
        tags_html = "".join(f'<span class="tag">↩ {name}</span>' for name, _ in top)
        st.markdown(
            f'<div style="margin:8px 0 0"><div class="section-title">Часті продукти</div>'
            f'{tags_html}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    # ── Quick Add from DB ──
    st.markdown('<div class="section-title">⚡ База продуктів (CSV)</div>', unsafe_allow_html=True)
    db = load_product_db()

    if db:
        cc1, cc2 = st.columns([2, 1])
        with cc1:
            cat = st.selectbox("Категорія", list(db.keys()), key="qa_cat",
                                label_visibility="collapsed")
        with cc2:
            qa_wt = st.number_input("Порція (г)", 1, 2000, 100, step=10,
                                     key="qa_wt", label_visibility="collapsed")

        items = db.get(cat, [])
        if items:
            bu_w = st.session_state.bu_weight
            rows = [{
                "Продукт":       p["name"],
                "Білки/100г":    p["protein"],
                "Ккал/100г":     int(p["calories"]),
                "Вугл./100г":    p["carbs"],
                f"ХО / {qa_wt}г": round((p["carbs"] * qa_wt / 100) / bu_w, 2),
            } for p in items]

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True, hide_index=True,
                column_config={
                    "Продукт": st.column_config.TextColumn(width="large"),
                    f"ХО / {qa_wt}г": st.column_config.NumberColumn(format="%.2f ⭐"),
                },
                height=220, key="db_table"
            )

            qa_c1, qa_c2 = st.columns([3, 1])
            with qa_c1:
                sel = st.selectbox("Оберіть для додавання",
                                    [p["name"] for p in items], key="qa_sel",
                                    label_visibility="collapsed")
            with qa_c2:
                if st.button("➕ Додати", key="btn_qa", use_container_width=True):
                    p_info = next(p for p in items if p["name"] == sel)
                    if add_product(sel, p_info["carbs"], qa_wt,
                                   p_info["protein"], p_info["calories"]):
                        st.rerun()

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    # ── Meal Templates ──
    with st.expander("📌 Шаблони прийомів їжі"):
        templates = st.session_state.meal_templates

        t_c1, t_c2 = st.columns(2)
        with t_c1:
            t_name = st.text_input("Назва шаблону", placeholder="Мій сніданок", key="t_name")
        with t_c2:
            st.markdown("<br/>", unsafe_allow_html=True)
            if st.button("💾 Зберегти поточний прийом як шаблон", key="btn_save_tpl"):
                if t_name and st.session_state.meal_data:
                    templates[t_name] = st.session_state.meal_data.copy()
                    save_data()
                    st.success(f"✅ Шаблон '{t_name}' збережено!")
                elif not t_name:
                    st.error("Введіть назву")
                else:
                    st.error("Додайте продукти спочатку")

        if templates:
            st.markdown('<div class="section-title" style="margin-top:8px">Збережені шаблони</div>', unsafe_allow_html=True)
            for name, items in templates.items():
                tbu = round(sum(i["ХО"] for i in items), 2)
                tca = round(sum(i["Вугл."] for i in items), 1)
                tc1, tc2, tc3 = st.columns([3, 1, 1])
                with tc1:
                    st.markdown(
                        f'<span class="badge badge-info" style="font-size:11px">{name}</span>'
                        f'<span style="font-size:11px;color:#6b7280;margin-left:8px">{tbu} ХО · {tca} г</span>',
                        unsafe_allow_html=True)
                with tc2:
                    if st.button("⚡ Завантажити", key=f"load_t_{name}"):
                        st.session_state.meal_data = [i.copy() for i in items]
                        save_data()
                        st.rerun()
                with tc3:
                    if st.button("🗑️", key=f"del_t_{name}"):
                        del templates[name]
                        save_data()
                        st.rerun()

    # ── Meal Table ──
    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    if not st.session_state.meal_data:
        st.markdown("""
        <div class="card" style="text-align:center;padding:48px;color:#4b5563;font-style:italic">
            🍽️ Прийом порожній. Додайте продукти вище.
        </div>""", unsafe_allow_html=True)
        return

    st.markdown('<div class="section-title">🍽️ Склад прийому</div>', unsafe_allow_html=True)
    df = pd.DataFrame(st.session_state.meal_data)

    st.dataframe(
        df[["Продукт","Вага","Вугл.","ХО","Ккал","ГН","Час"]],
        use_container_width=True, hide_index=True,
        column_config={
            "Продукт": st.column_config.TextColumn(width="large"),
            "Вага":    st.column_config.NumberColumn("Вага (г)", format="%d г"),
            "Вугл.":   st.column_config.NumberColumn("Вуглеводи", format="%.1f г"),
            "ХО":      st.column_config.NumberColumn("ХО", format="%.2f ⭐"),
            "Ккал":    st.column_config.NumberColumn("Ккал", format="%.0f"),
            "ГН":      st.column_config.NumberColumn("Глік. навант.", format="%.1f"),
        },
        key="meal_df"
    )

    # Remove + export row
    if len(st.session_state.meal_data) > 1:
        rc1, rc2, rc3 = st.columns([3, 1, 1])
        with rc1:
            to_rm = st.selectbox("Видалити продукт", range(len(st.session_state.meal_data)),
                format_func=lambda i: f"{st.session_state.meal_data[i]['Продукт']} "
                                      f"({st.session_state.meal_data[i]['ХО']} ХО)",
                key="rm_sel", label_visibility="collapsed")
        with rc2:
            if st.button("🗑️ Видалити", key="btn_rm"):
                removed = st.session_state.meal_data.pop(to_rm)
                save_data()
                st.success(f"Видалено: {removed['Продукт']}")
                st.rerun()
        with rc3:
            csv_data = pd.DataFrame(st.session_state.meal_data).to_csv(index=False)
            st.download_button("📥 CSV", data=csv_data,
                file_name=f"meal_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", key="dl_csv")

    # Results card
    t = get_totals()
    st.markdown(result_card(t["bu"], t["carbs"], t["cal"], t["gl"]),
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — INSULIN CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def tab_insulin():
    st.markdown("""
    <div class="warn-bar">
      ⚠️ Тільки для ознайомлення. Будь-яке коригування доз — виключно з ендокринологом!
    </div>""", unsafe_allow_html=True)

    ip = st.session_state.insulin_profile

    # ── Parameters ──
    st.markdown('<div class="section-title">📐 Параметри інсуліну</div>', unsafe_allow_html=True)
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        cr  = st.number_input("🍞 CR — г вуглеводів на 1 ОД",
                               5, 30, int(ip["cr"]), step=1, key="ins_cr")
    with ic2:
        isf = st.number_input("📉 ISF — ммоль/л на 1 ОД",
                               0.5, 8.0, float(ip["isf"]), step=0.1, key="ins_isf")
    with ic3:
        iob_dur = st.number_input("⏳ Тривалість дії (год)",
                                   2.0, 8.0, float(ip["iob_duration"]),
                                   step=0.5, key="ins_dur")

    # Save if changed
    if cr != ip["cr"] or isf != ip["isf"] or iob_dur != ip["iob_duration"]:
        ip.update({"cr": cr, "isf": isf, "iob_duration": iob_dur})
        save_data()

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    # ── Input ──
    st.markdown('<div class="section-title">📊 Вхідні дані</div>', unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)
    with ic1:
        g_cur = st.number_input("🩸 Поточна глюкоза (ммоль/л)",
                                  0.5, 35.0, 5.5, step=0.1, key="ins_gcur")
    with ic2:
        g_tgt = st.number_input("🎯 Цільова глюкоза (ммоль/л)",
                                  3.0, 12.0,
                                  st.session_state.user_profile["target_min"],
                                  step=0.1, key="ins_gtgt")

    label, key_z, color, msg = get_zone(g_cur)
    badge_cls = f"badge-{key_z}" if key_z != "unknown" else "badge-info"
    st.markdown(f'<span class="badge {badge_cls}">{label}</span>'
                f'<span style="font-size:12px;color:#6b7280;margin-left:8px">{msg}</span>',
                unsafe_allow_html=True)

    # IOB from recent doses
    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⏳ Активний інсулін (IOB)</div>', unsafe_allow_html=True)

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        iob_units = st.number_input("Остання доза (ОД)", 0.0, 30.0, 0.0, step=0.5, key="iob_u")
    with ic2:
        iob_min   = st.number_input("Хвилин тому", 0, 480, 0, step=5, key="iob_m")
    with ic3:
        computed_iob = iob_remaining(iob_units, iob_min, iob_dur)
        st.markdown(f"""
        <div class="iob-bar">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:11px;color:#6b7280">Залишок IOB</span>
            <span style="font-family:'DM Mono',monospace;color:#60a5fa;font-size:13px">{computed_iob} ОД</span>
          </div>
          <div style="background:rgba(255,255,255,.05);border-radius:6px;height:5px;overflow:hidden">
            <div style="background:linear-gradient(90deg,#60a5fa,#e8506a);
                        width:{min(100, computed_iob/max(iob_units,0.1)*100):.0f}%;
                        height:100%;border-radius:6px;transition:width 0.5s"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # TOD multiplier
    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    tod_key = st.radio("🕐 Час доби (впливає на ISF)", list(TOD_ISF_FACTORS.keys()),
                        horizontal=True, key="ins_tod")
    tod_factor = TOD_ISF_FACTORS[tod_key]
    adj_isf    = isf * tod_factor
    st.caption(f"ISF скоригований: {isf:.1f} × {tod_factor:.2f} = {adj_isf:.2f} ммоль/л на ОД")

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    # ── Calculate ──
    totals = get_totals()
    carbs  = totals["carbs"]

    if st.button("⚡ Розрахувати дозу болюсу", use_container_width=True, key="ins_calc"):
        meal_dose = carbs / cr if cr > 0 else 0
        corr_dose = (g_cur - g_tgt) / adj_isf if adj_isf > 0 else 0
        total     = max(0.0, meal_dose + corr_dose - computed_iob)
        rounded   = round(total * 2) / 2

        st.markdown(dose_card(total, rounded, meal_dose, corr_dose,
                               computed_iob, carbs, tod_factor),
                    unsafe_allow_html=True)

        # Alerts
        if g_cur < 4.0:
            st.error("🚨 ГІПОГЛІКЕМІЯ! НЕ вводьте інсулін! Спочатку з'їжте 15 г швидких вуглеводів (сік, глюкоза). Перевірте глюкозу через 15 хвилин.")
        elif g_cur >= 14.0:
            st.warning("⚠️ Дуже високий рівень! Перевірте кетони у крові. При наявності кетонів — зверніться до лікаря!")
        elif g_cur >= 10.0:
            st.warning("⚠️ Підвищений рівень. Перевірте кетони.")

        # Glucose prediction chart
        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Прогноз глюкози після болюсу (2 год)</div>', unsafe_allow_html=True)

        times, pred = glucose_prediction_curve(g_cur, total, carbs, adj_isf, cr)
        target_min  = st.session_state.user_profile["target_min"]
        target_max  = st.session_state.user_profile["target_max"]

        fig = go.Figure()
        fig.add_hrect(y0=target_min, y1=target_max,
                      fillcolor="rgba(52,211,153,0.06)", line_width=0,
                      annotation_text="Цільова зона", annotation_font_color="#34d399",
                      annotation_font_size=10, annotation_position="top left")
        fig.add_trace(go.Scatter(
            x=times, y=pred, mode="lines",
            line=dict(color="#e8506a", width=2.5, shape="spline"),
            fill="tozeroy", fillcolor="rgba(232,80,106,0.06)",
            name="Прогноз глюкози",
            hovertemplate="<b>%{x} хв</b><br>%{y:.1f} ммоль/л<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=[0], y=[g_cur], mode="markers",
            marker=dict(size=10, color="#fbbf24",
                        line=dict(color="#080a0e", width=2)),
            name="Зараз", hovertemplate="Зараз: %{y:.1f}<extra></extra>"
        ))
        fig.update_layout(
            height=280, showlegend=False,
            xaxis_title="Хвилини після болюсу",
            yaxis_title="ммоль/л",
            **PLOTLY_THEME
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚠️ Прогноз є спрощеною моделлю. Реальний результат залежить від багатьох факторів.")

    else:
        st.info(f"📋 Вуглеводів у поточному прийомі: **{carbs} г ({totals['bu']} ХО)**. Натисніть «Розрахувати».")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def tab_analytics():
    glogs = st.session_state.glucose_logs
    dtots = st.session_state.daily_totals

    if not glogs and not dtots:
        st.markdown("""
        <div class="card" style="text-align:center;padding:56px;color:#4b5563">
          <div style="font-size:2rem;margin-bottom:12px">📊</div>
          <div>Почніть записувати глюкозу та зберігайте прийоми їжі,<br>щоб тут з'явилася аналітика.</div>
        </div>""", unsafe_allow_html=True)
        return

    a1, a2 = st.tabs(["🩸 Глюкоза & TIR", "🍽️ Прийоми їжі"])

    with a1:
        _analytics_glucose(glogs)

    with a2:
        _analytics_meals(dtots)


def _analytics_glucose(logs: list):
    if not logs:
        st.info("Додайте вимірювання глюкози через бічну панель")
        return

    stats = tir_stats(logs)

    # ── HbA1c card ──
    st.markdown(hba1c_card(stats["hba1c"], stats["avg"], stats["target"]),
                unsafe_allow_html=True)

    c1, c2 = st.columns([1.6, 1])

    with c1:
        # Glucose timeline
        times  = [e["timestamp"][:16].replace("T", " ") for e in logs[-40:]]
        levels = [e["level"] for e in logs[-40:]]
        colors = [get_zone(lv)[2] for lv in levels]
        tmin   = st.session_state.user_profile["target_min"]
        tmax   = st.session_state.user_profile["target_max"]

        fig = go.Figure()
        fig.add_hrect(y0=tmin, y1=tmax,
                      fillcolor="rgba(52,211,153,0.05)", line_width=0,
                      annotation_text=f"Ціль {tmin}–{tmax}", annotation_font_size=10,
                      annotation_font_color="#34d399")
        fig.add_trace(go.Scatter(
            x=times, y=levels, mode="lines+markers",
            line=dict(color="#60a5fa", width=2, shape="spline"),
            marker=dict(size=8, color=colors,
                        line=dict(color="#080a0e", width=1.5)),
            hovertemplate="<b>%{x}</b><br>%{y:.1f} ммоль/л<extra></extra>"
        ))
        fig.update_layout(
            title="🩸 Глюкоза в часі", height=300,
            xaxis=dict(tickangle=-35, nticks=8),
            **PLOTLY_THEME
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # TIR donut
        tir_vals   = [stats["hypo"], stats["low"], stats["target"],
                      stats["high"], stats["very_high"]]
        tir_labels = ["Гіпо <4.0", "Низький 4–4.5",
                      "Норма 4.5–7.8", "Висок. 7.8–14", "Дуже висок. >14"]
        tir_colors = ["#f87171","#fb923c","#34d399","#fbbf24","#ef4444"]

        fig2 = go.Figure(go.Pie(
            labels=tir_labels, values=tir_vals, hole=0.60,
            marker=dict(colors=tir_colors,
                        line=dict(color="#080a0e", width=2)),
            textinfo="percent", textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>%{value}%<extra></extra>"
        ))
        fig2.add_annotation(
            text=f"<b>{stats['target']}%</b><br>в нормі",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#34d399", family="DM Mono")
        )
        fig2.update_layout(
            title="🎯 Time-in-Range", height=300,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9ca3af"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # TIR legend
        legend_html = '<div class="tir-legend">'
        for label, color, val in zip(tir_labels, tir_colors, tir_vals):
            legend_html += f'<div class="tir-item"><div class="tir-dot" style="background:{color}"></div>{label}: {val}%</div>'
        legend_html += '</div>'
        st.markdown(legend_html, unsafe_allow_html=True)

    # ── Glucose distribution histogram ──
    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    levels_all = [e["level"] for e in logs]

    with c3:
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=levels_all, nbinsx=20,
            marker_color="#60a5fa",
            marker_line=dict(color="#080a0e", width=1),
            hovertemplate="<b>%{x:.1f} мм</b><br>%{y} вимірювань<extra></extra>"
        ))
        fig3.add_vrect(x0=tmin, x1=tmax, fillcolor="rgba(52,211,153,0.07)",
                       line_width=0, annotation_text="Норма",
                       annotation_font_color="#34d399", annotation_font_size=10)
        fig3.update_layout(
            title="📊 Розподіл глюкози", height=260,
            xaxis_title="ммоль/л", yaxis_title="Вимірювань",
            **PLOTLY_THEME
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # Stats metrics
        st.metric("Всього вимірювань", stats["n"])
        st.metric("Середня глюкоза", f"{stats['avg']} ммоль/л",
                  delta=f"{'↑' if stats['avg'] > tmax else ('✓' if stats['avg'] >= tmin else '↓')}")
        st.metric("Стд. відхилення", f"{stats['std']} ммоль/л",
                  delta="добре" if stats["std"] < 2.5 else "варіабельно",
                  delta_color="normal" if stats["std"] < 2.5 else "inverse")
        st.metric("Розрах. HbA1c", f"{stats['hba1c']}%")


def _analytics_meals(dtots: dict):
    if not dtots:
        st.info("Збережіть кілька прийомів їжі, щоб побачити аналітику")
        return

    # Build time series
    sorted_dates = sorted(dtots.keys())[-30:]
    dates, bus, carbss, meal_counts = [], [], [], []
    for d in sorted_dates:
        meals = dtots[d]
        dates.append(d)
        bus.append(round(sum(m["totals"]["bu"] for m in meals), 1))
        carbss.append(round(sum(m["totals"]["carbs"] for m in meals), 1))
        meal_counts.append(len(meals))

    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=bus, mode="lines+markers", name="ХО/день",
            line=dict(color="#e8506a", width=2, shape="spline"),
            marker=dict(size=6, color="#e8506a", line=dict(color="#080a0e",width=1.5)),
            fill="tozeroy", fillcolor="rgba(232,80,106,0.06)",
            hovertemplate="<b>%{x}</b><br>%{y:.1f} ХО<extra></extra>"
        ))
        fig.update_layout(title="📈 ХО по днях", height=260, **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(
            x=dates, y=carbss, name="Вуглеводи г",
            marker_color="#f97b4f", marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>%{y:.0f} г<extra></extra>"
        ))
        fig2.add_trace(go.Scatter(
            x=dates, y=meal_counts, name="Прийомів",
            mode="lines+markers",
            line=dict(color="#60a5fa", width=2),
            marker=dict(size=5),
            hovertemplate="<b>%{x}</b><br>%{y} прийомів<extra></extra>"
        ), secondary_y=True)
        fig2.update_layout(
            title="🍞 Вуглеводи + прийоми", height=260,
            legend=dict(orientation="h", y=-0.15, font=dict(size=10, color="#9ca3af")),
            **PLOTLY_THEME
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Meal patterns heatmap ──
    patterns = st.session_state.meal_patterns
    if patterns:
        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)

        with c3:
            meal_names = list(patterns.keys())
            avg_bus    = [round(sum(m["bu"] for m in v) / len(v), 1)
                          for v in patterns.values()]
            counts     = [len(v) for v in patterns.values()]
            fig3 = go.Figure(go.Bar(
                x=meal_names, y=avg_bus,
                marker_color=["#34d399","#60a5fa","#fbbf24","#e8506a"][:len(meal_names)],
                marker_line_width=0,
                text=[f"{b} ХО<br>({c}x)" for b,c in zip(avg_bus, counts)],
                textposition="outside", textfont=dict(color="#9ca3af", size=10),
                hovertemplate="<b>%{x}</b><br>Середньо %{y:.1f} ХО<extra></extra>"
            ))
            fig3.update_layout(title="🕐 Середнє ХО по типу прийому",
                               height=280, **PLOTLY_THEME)
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            # Top products by frequency
            freq = st.session_state.product_freq
            if freq:
                top10 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
                fig4  = go.Figure(go.Bar(
                    x=[v for _, v in top10],
                    y=[n for n, _ in top10],
                    orientation="h",
                    marker_color="#8b5cf6", marker_line_width=0,
                    hovertemplate="<b>%{y}</b><br>%{x}x<extra></extra>"
                ))
                fig4.update_layout(title="🏆 Топ продуктів",
                                   height=280, **PLOTLY_THEME,
                                   xaxis_title="Разів")
                st.plotly_chart(fig4, use_container_width=True)

    # Weekly stats
    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📅 Зведення за останній тиждень</div>', unsafe_allow_html=True)
    _weekly_stats_row()


def _weekly_stats_row():
    end   = datetime.now()
    start = end - timedelta(days=7)
    week  = {d: ms for d, ms in st.session_state.daily_totals.items()
             if start <= datetime.fromisoformat(d) <= end}

    if not week:
        st.info("Немає даних за останні 7 днів")
        return

    total_m = sum(len(ms) for ms in week.values())
    all_bu  = [sum(m["totals"]["bu"] for m in ms) for ms in week.values()]
    avg_bu  = round(sum(all_bu) / len(all_bu), 1)
    max_bu  = round(max(all_bu), 1)
    min_bu  = round(min(all_bu), 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Прийомів", total_m)
    c2.metric("Середньо ХО/день", avg_bu)
    c3.metric("Макс ХО/день", max_bu)
    c4.metric("Мін ХО/день", min_bu)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — HISTORY & REPORTS
# ══════════════════════════════════════════════════════════════════════════════

def tab_history():
    dtots = st.session_state.daily_totals

    if not dtots:
        st.markdown("""
        <div class="card" style="text-align:center;padding:56px;color:#4b5563">
          <div style="font-size:2rem;margin-bottom:12px">📋</div>
          Збережіть перший прийом їжі, щоб він з'явився тут.
        </div>""", unsafe_allow_html=True)
        return

    # Date filter
    all_dates = sorted(dtots.keys(), reverse=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        show_days = st.slider("Показати днів", 3, min(60, len(all_dates)), 14, key="hist_days")
    with c2:
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("📥 Експорт JSON", key="exp_json"):
            j = json.dumps({
                "profile":      st.session_state.user_profile,
                "daily_totals": st.session_state.daily_totals,
                "glucose_logs": st.session_state.glucose_logs,
                "exported":     datetime.now().isoformat(),
            }, ensure_ascii=False, indent=2)
            st.download_button("⬇️ Завантажити", data=j,
                file_name=f"glypro_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json", key="dl_json")

    for date in all_dates[:show_days]:
        meals    = dtots[date]
        day_bu   = round(sum(m["totals"]["bu"]    for m in meals), 1)
        day_carbs= round(sum(m["totals"]["carbs"] for m in meals), 1)
        day_cal  = round(sum(m["totals"]["cal"]   for m in meals), 0)

        with st.expander(f"📅  {date}  ·  {len(meals)} прийомів  ·  {day_bu} ХО  ·  {day_carbs} г вугл."):
            for meal in meals:
                ts  = meal["timestamp"][11:16]
                mt  = meal.get("meal_type", "")
                t   = meal["totals"]
                n_p = len(meal.get("data", []))

                st.markdown(f"""
                <div class="card card-sm" style="margin-bottom:8px">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                      <span style="font-size:11px;color:#4b5563">🕐 {ts} &nbsp; {mt}</span>
                      <span style="font-size:11px;color:#6b7280;margin-left:12px">{n_p} продуктів</span>
                    </div>
                    <div style="text-align:right">
                      <span style="font-family:'DM Mono',monospace;color:#e8506a;font-size:1rem;font-weight:500">{t["bu"]} ХО</span>
                      <span style="font-size:11px;color:#6b7280;margin-left:8px">{t["carbs"]} г · {int(t["cal"])} ккал</span>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

                if meal.get("data"):
                    df = pd.DataFrame(meal["data"])
                    cols = [c for c in ["Продукт","Вага","Вугл.","ХО","Ккал"] if c in df.columns]
                    st.dataframe(df[cols], use_container_width=True,
                                 hide_index=True, height=150,
                                 key=f"hist_{date}_{ts}")

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    if st.checkbox("🗑️ Показати кнопку очищення журналу"):
        if st.button("❌ Очистити весь журнал", key="clr_hist_btn"):
            st.session_state.daily_totals = {}
            save_data()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

def tab_settings():
    st.markdown('<div class="section-title">👤 Профіль</div>', unsafe_allow_html=True)
    p = st.session_state.user_profile

    with st.form("profile_form"):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            name   = st.text_input("Ім'я", p.get("name",""))
            age    = st.number_input("Вік", 1, 120, p.get("age", 25))
        with sc2:
            weight = st.number_input("Вага (кг)", 20, 200, p.get("weight", 60))
            height = st.number_input("Зріст (см)", 100, 250, p.get("height", 165))
        with sc3:
            activity    = st.selectbox("Активність",
                ["low","medium","high"],
                index=["low","medium","high"].index(p.get("activity","medium")))
            insulin_type = st.selectbox("Тип інсуліну",
                ["rapid","short","intermediate","long"],
                index=["rapid","short","intermediate","long"]
                      .index(p.get("insulin_type","rapid")))

        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 Цільові показники</div>', unsafe_allow_html=True)
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            t_min = st.number_input("Ціль глюкози мін (ммоль/л)", 3.0, 8.0,
                                    float(p.get("target_min", 4.5)), step=0.1)
        with tc2:
            t_max = st.number_input("Ціль глюкози макс (ммоль/л)", 5.0, 15.0,
                                    float(p.get("target_max", 7.8)), step=0.1)
        with tc3:
            tdd   = st.number_input("TDD — добова доза (ОД)", 5.0, 300.0,
                                    float(p.get("tdd", 35.0)), step=0.5)

        submitted = st.form_submit_button("💾 Зберегти профіль", use_container_width=True)
        if submitted:
            st.session_state.user_profile.update({
                "name": name, "age": age, "weight": weight, "height": height,
                "activity": activity, "insulin_type": insulin_type,
                "target_min": t_min, "target_max": t_max, "tdd": tdd,
            })
            # Auto-update insulin params from TDD
            cr_auto  = round(500 / tdd, 1)
            isf_auto = round(1700 / tdd / 18, 2)
            st.session_state.insulin_profile.update({"cr": cr_auto, "isf": isf_auto})
            save_data()
            st.success(f"✅ Профіль збережено! Автоматичні параметри: CR = {cr_auto} г/ОД, ISF = {isf_auto} ммоль/ОД")

    # ── Auto-calculated params ──
    tdd = st.session_state.user_profile.get("tdd", 35.0)
    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧮 Автоматично розраховані параметри (правило 500/1700)</div>', unsafe_allow_html=True)
    ac1, ac2, ac3, ac4 = st.columns(4)
    ac1.metric("CR (500 / TDD)", f"{round(500/tdd,1)} г/ОД",
               help="Г вуглеводів на 1 ОД болюсного інсуліну")
    ac2.metric("ISF (1700 / TDD / 18)", f"{round(1700/tdd/18,2)} мм/ОД",
               help="На скільки ммоль/л знижує 1 ОД інсуліну")
    ac3.metric("Базальна доза", f"{round(tdd*0.5,1)} ОД/добу",
               help="~50% від TDD — базальний інсулін")
    ac4.metric("Болюсна доза", f"{round(tdd*0.5,1)} ОД/добу",
               help="~50% від TDD — болюсний інсулін")

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    # ── Danger zone ──
    with st.expander("⚠️ Небезпечна зона — видалення даних"):
        st.warning("Ці дії незворотні!")
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            if st.button("🗑️ Очистити поточний прийом"):
                st.session_state.meal_data = []
                save_data()
                st.success("✅ Прийом очищено")
        with dc2:
            if st.button("🗑️ Очистити глюкозу"):
                st.session_state.glucose_logs = []
                save_data()
                st.success("✅ Журнал глюкози очищено")
        with dc3:
            if st.button("💥 Скинути ВСЕ"):
                for key in ["meal_data","daily_totals","glucose_logs",
                            "meal_patterns","meal_templates","product_freq",
                            "product_history"]:
                    st.session_state[key] = [] if isinstance(
                        st.session_state[key], list) else {}
                save_data()
                st.success("✅ Усі дані скинуто")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="GlyPro",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session()
    inject_css()
    render_sidebar()

    # ── Header ──
    hc1, hc2 = st.columns([3, 2])
    with hc1:
        name_greeting = f", {st.session_state.user_profile['name']}" \
                        if st.session_state.user_profile.get("name") else ""
        st.markdown(
            f'<div class="glyco-title" style="font-size:2.6rem">GlyPro</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:13px;color:#4b5563;margin-top:2px">'
            f'Щоденник діабетика 1 типу{name_greeting} · v{APP_VERSION}</div>',
            unsafe_allow_html=True)
    with hc2:
        # Quick status summary
        t = get_totals()
        glogs = st.session_state.glucose_logs
        last_g = glogs[-1]["level"] if glogs else None
        st.markdown(f"""
        <div style="display:flex;gap:10px;justify-content:flex-end;align-items:center;margin-top:8px">
          {f'<span class="badge badge-{get_zone(last_g)[1]}">{last_g} мм</span>' if last_g else ''}
          {f'<span class="badge badge-info">{t["bu"]} ХО</span>' if t["bu"] > 0 else ''}
          {f'<span class="badge badge-info">{len(st.session_state.meal_data)} продуктів</span>' if st.session_state.meal_data else ''}
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider" style="margin:12px 0 20px"/>', unsafe_allow_html=True)

    # ── Navigation tabs ──
    t1, t2, t3, t4, t5 = st.tabs([
        "🍽️ Прийом їжі",
        "💉 Інсулін",
        "📊 Аналітика",
        "📋 Журнал",
        "⚙️ Налаштування",
    ])

    with t1: tab_meal()
    with t2: tab_insulin()
    with t3: tab_analytics()
    with t4: tab_history()
    with t5: tab_settings()

    # ── Footer ──
    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    total_days = len(st.session_state.daily_totals)
    total_g    = len(st.session_state.glucose_logs)
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                font-size:11px;color:#374151;padding-bottom:8px;flex-wrap:wrap;gap:8px">
      <div>
        GlyPro v{APP_VERSION} ·
        <span style="color:#e8506a">♥</span> Зроблено з турботою ·
        <span style="color:#4b5563">Тільки для ознайомлення</span>
      </div>
      <div style="display:flex;gap:12px;color:#4b5563">
        <span>📅 {total_days} днів</span>
        <span>🩸 {total_g} вимірювань</span>
        <span>🍽️ {len(st.session_state.product_history)} продуктів у базі</span>
      </div>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

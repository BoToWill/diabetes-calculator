"""
GlyPro v7.0 — Щоденник діабетика 1 типу
⚠️  Тільки для ознайомлення. Консультуйтеся з ендокринологом.

NEW in v7.0:
  • Lite / Pro режим — перемикач у сайдбарі
  • Темна / світла тема (по стандарту: темна)
  • Lite — мінімальний, фокусований інтерфейс
  • Pro — повний функціонал (всі вкладки)
"""

import streamlit as st
import pandas as pd
import json
import os
import math
import re
import requests
from datetime import datetime, timedelta
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

APP_VERSION    = "7.1"
GEMINI_API_KEY = "AIzaSyAliac_9z_02hLWs0PxeWRj3CFx32JzH_Q"
GEMINI_URL     = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY
)
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

KETONE_ZONES = [
    (0.0,  0.5,  "✅ Норма",        "#34d399", "Кетони в нормі"),
    (0.6,  1.5,  "🟡 Помірні",      "#fbbf24", "Помірний кетоз. Випийте воду, перевірте глюкозу."),
    (1.6,  3.0,  "🟠 Підвищені",    "#fb923c", "Підвищені кетони! Зверніться до лікаря якнайшвидше."),
    (3.1,  99.0, "🔴 НЕБЕЗПЕКА",    "#f87171", "ДКА ризик! Негайно до лікарні!"),
]

TOD_ISF_FACTORS = {
    "🌅 Ранок (6–10)":   0.85,
    "☀️ День (10–17)":   1.00,
    "🌆 Вечір (17–21)":  1.05,
    "🌙 Ніч (21–6)":     1.10,
}


# ══════════════════════════════════════════════════════════════════════════════
#  PURE MATH / MEDICAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_zone(g: float) -> tuple:
    for lo, hi, label, key, color, msg in GLUCOSE_ZONES:
        if lo <= g <= hi:
            return label, key, color, msg
    return "❓", "unknown", "#6b7280", ""


def get_ketone_zone(k: float) -> tuple:
    for lo, hi, label, color, msg in KETONE_ZONES:
        if lo <= k <= hi:
            return label, color, msg
    return "❓", "#6b7280", ""


def calc_bu(carbs100: float, weight_g: float, bu_w: float) -> dict:
    carbs = (carbs100 * weight_g) / 100
    return {
        "carbs":    round(carbs, 1),
        "bu":       round(carbs / bu_w, 2),
        "calories": round(carbs * 4, 0),
        "gl":       round(carbs * 50 / 100, 1),
    }


def iob_remaining(units: float, minutes_ago: float, duration_h: float = 4.0) -> float:
    if minutes_ago <= 0 or units <= 0:
        return units
    t   = minutes_ago / 60
    dur = duration_h
    if t >= dur:
        return 0.0
    pct = 1 - (t / dur) ** 2 * (3 - 2 * t / dur)
    return max(0.0, round(units * pct, 2))


def total_iob(active_doses: list, iob_duration: float) -> float:
    now = datetime.now()
    total = 0.0
    for dose in active_doses:
        ts  = datetime.fromisoformat(dose["timestamp"])
        min_ago = (now - ts).total_seconds() / 60
        total  += iob_remaining(dose["units"], min_ago, iob_duration)
    return round(total, 2)


def estimate_hba1c(avg_glucose_mmol: float) -> float:
    avg_mg = avg_glucose_mmol * 18.0
    return round((avg_mg + 46.7) / 28.7, 1)


def estimate_gmi(avg_glucose_mmol: float) -> float:
    avg_mg = avg_glucose_mmol * 18.0
    return round(3.31 + 0.02392 * avg_mg, 1)


def calc_cv(levels: list) -> float:
    if len(levels) < 2:
        return 0.0
    mean = np.mean(levels)
    if mean == 0:
        return 0.0
    return round((np.std(levels) / mean) * 100, 1)


def glucose_trend_arrow(logs: list) -> str:
    if len(logs) < 2:
        return "→"
    recent = logs[-6:]
    if len(recent) < 2:
        return "→"
    try:
        t1 = datetime.fromisoformat(recent[-2]["timestamp"])
        t2 = datetime.fromisoformat(recent[-1]["timestamp"])
        g1 = recent[-2]["level"]
        g2 = recent[-1]["level"]
        dt_hours = max((t2 - t1).total_seconds() / 3600, 0.01)
        roc = (g2 - g1) / dt_hours
        if   roc >  2.2: return "↑↑"
        elif roc >  1.1: return "↑"
        elif roc > -1.1: return "→"
        elif roc > -2.2: return "↓"
        else:             return "↓↓"
    except Exception:
        return "→"


def glucose_prediction_curve(current_g: float, dose_units: float,
                              carbs_g: float, isf: float, cr: float,
                              minutes: int = 240) -> tuple:
    times, values = [], []
    for t in range(0, minutes + 1, 10):
        carb_peak = 45
        if t <= carb_peak:
            absorbed_pct = t / carb_peak
        else:
            absorbed_pct = max(0, 1 - (t - carb_peak) / 135)
        glucose_from_carbs = (carbs_g * absorbed_pct * 0.55) / 18
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


def extended_bolus(carbs_g: float, protein_g: float, fat_g: float, cr: float) -> dict:
    carb_dose = carbs_g / cr if cr > 0 else 0
    prot_eq   = protein_g * 0.5
    fat_eq    = fat_g * 0.1
    ext_carbs = prot_eq + fat_eq
    ext_dose  = ext_carbs / cr if cr > 0 else 0
    total     = carb_dose + ext_dose
    return {
        "carb_dose":  round(carb_dose, 2),
        "ext_dose":   round(ext_dose, 2),
        "total":      round(total, 2),
        "rounded":    round(round(total * 2) / 2, 1),
        "prot_eq":    round(prot_eq, 1),
        "fat_eq":     round(fat_eq, 1),
        "ext_carbs":  round(ext_carbs, 1),
        "normal_pct": round(carb_dose / total * 100) if total > 0 else 100,
        "ext_pct":    round(ext_dose  / total * 100) if total > 0 else 0,
    }


def get_meal_type(hour: int) -> str:
    if  5 <= hour < 11: return "🌅 Сніданок"
    if 11 <= hour < 15: return "☀️ Обід"
    if 15 <= hour < 18: return "🍵 Перекус"
    return "🌙 Вечеря"


def auto_tod_key() -> str:
    h = datetime.now().hour
    if  6 <= h < 10: return "🌅 Ранок (6–10)"
    if 10 <= h < 17: return "☀️ День (10–17)"
    if 17 <= h < 21: return "🌆 Вечір (17–21)"
    return "🌙 Ніч (21–6)"


def tir_stats(logs: list) -> dict:
    if not logs:
        return {"hypo": 0, "low": 0, "target": 0, "high": 0, "very_high": 0,
                "avg": 0, "n": 0, "std": 0, "cv": 0, "gmi": 0, "hba1c": 0}
    levels = [e["level"] for e in logs]
    n = len(levels)
    avg = np.mean(levels)
    return {
        "hypo":      round(sum(1 for g in levels if g < 4.0)  / n * 100),
        "low":       round(sum(1 for g in levels if 4.0 <= g < 4.5) / n * 100),
        "target":    round(sum(1 for g in levels if 4.5 <= g <= 7.8) / n * 100),
        "high":      round(sum(1 for g in levels if 7.8 < g <= 13.9) / n * 100),
        "very_high": round(sum(1 for g in levels if g > 13.9) / n * 100),
        "avg":       round(avg, 1),
        "std":       round(np.std(levels), 1),
        "cv":        calc_cv(levels),
        "n":         n,
        "hba1c":     estimate_hba1c(avg),
        "gmi":       estimate_gmi(avg),
    }


def calc_streak(glucose_logs: list, target_min: float, target_max: float) -> int:
    from collections import defaultdict
    by_date = defaultdict(list)
    for e in glucose_logs:
        date = e.get("date", e["timestamp"][:10])
        by_date[date].append(e["level"])
    streak = 0
    today  = datetime.now().date()
    for i in range(365):
        d = str(today - timedelta(days=i))
        if d not in by_date:
            break
        levels = by_date[d]
        tir = sum(1 for g in levels if target_min <= g <= target_max) / len(levels)
        if tir >= 0.70:
            streak += 1
        else:
            break
    return streak


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
                "ketone_logs":     ss.ketone_logs,
                "dose_log":        ss.dose_log,
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
#  GEMINI AI COMMAND ENGINE
# ══════════════════════════════════════════════════════════════════════════════

AI_SYSTEM_PROMPT = """Ти — розумний асистент медичного щоденника GlyPro для людей з діабетом 1 типу.
Твоя задача: розпізнати команду користувача і повернути JSON-об'єкт (і ТІЛЬКИ JSON, без markdown, без зайвого тексту).

=== ДОСТУПНІ ДІЇ ===

1. Записати глюкозу:
{"action":"log_glucose","level":6.2,"time":"Перед їжею","message":"✅ Глюкоза 6.2 ммоль/л записана"}

2. Додати їжу вручну:
{"action":"add_food","name":"Гречка варена","carbs100":21.0,"weight":200,"message":"✅ Гречка 200г додана до прийому"}

3. Розрахувати болюс (вказати поточну глюкозу):
{"action":"calc_bolus","glucose":9.5,"message":"⚡ Розраховую дозу для глюкози 9.5..."}

4. Записати дозу інсуліну:
{"action":"log_dose","units":4.0,"note":"Сніданок","message":"✅ 4.0 ОД інсуліну записано"}

5. Записати кетони:
{"action":"log_ketones","value":0.3,"message":"✅ Кетони 0.3 ммоль/л записані"}

6. Зберегти поточний прийом:
{"action":"save_meal","message":"💾 Прийом збережено в журнал"}

7. Очистити поточний прийом:
{"action":"clear_meal","message":"🗑️ Прийом очищено"}

8. Показати статистику / відповісти на питання:
{"action":"answer","message":"<повна відповідь текстом>"}

=== КОНТЕКСТ ПАЦІЄНТА (підставляється динамічно) ===
{context}

=== ПРАВИЛА ===
- Якщо команда стосується їжі без точних даних про вуглеводи — зроби розумне припущення на основі загальновідомих значень
- Завжди відповідай ТІЛЬКИ валідним JSON без ```json та без пояснень
- Якщо не зрозуміло — використай action:"answer" з поясненням
- Числа завжди float або int (не рядки)
- Час для глюкози: "Перед їжею" | "Після їжі (1 год)" | "Після їжі (2 год)" | "Вранці натще" | "Перед сном"
- При небезпечних рівнях глюкози (<3.9 або >14) — обов'язково попередь у message
- Відповідай мовою користувача (українська або та, якою написали)
"""


def _build_ai_context() -> str:
    """Build current app state context for AI."""
    ss = st.session_state
    glogs = ss.glucose_logs
    last_g = glogs[-1]["level"] if glogs else None
    trend  = glucose_trend_arrow(glogs)
    t      = get_totals()
    ip     = ss.insulin_profile
    p      = ss.user_profile
    tiob   = total_iob(ip.get("active_doses", []), ip.get("iob_duration", 4.0))

    meal_items = ""
    if ss.meal_data:
        meal_items = ", ".join(
            f"{i['Продукт']} {i['Вага']}г ({i['ХО']} ХО)" for i in ss.meal_data[-5:]
        )

    recent_glucose = ""
    if glogs:
        recent_glucose = " → ".join(
            f"{e['level']} мм ({e['timestamp'][11:16]})" for e in glogs[-4:]
        )

    return f"""
Поточний час: {datetime.now().strftime('%H:%M %d.%m.%Y')}
Остання глюкоза: {last_g} ммоль/л (тренд: {trend})
Останні вимірювання: {recent_glucose}
Поточний прийом їжі: {meal_items or 'порожній'}
Всього в прийомі: {t['carbs']} г вуглеводів, {t['bu']} ХО, {int(t['cal'])} ккал
CR (вуглеводний коефіцієнт): {ip.get('cr', 10)} г/ОД
ISF (чутливість до інсуліну): {ip.get('isf', 2.5)} ммоль/ОД
Активний IOB: {tiob} ОД
Цільова глюкоза: {p.get('target_min', 4.5)}–{p.get('target_max', 7.8)} ммоль/л
TDD (добова доза): {p.get('tdd', 35)} ОД
Вага пацієнта: {p.get('weight', 60)} кг
"""


def _get_available_gemini_model() -> str | None:
    """Fetch available models from API and return first that supports generateContent."""
    for api_ver in ("v1beta", "v1"):
        url = (
            f"https://generativelanguage.googleapis.com/{api_ver}/models"
            f"?key={GEMINI_API_KEY}"
        )
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            data   = resp.json()
            models = data.get("models", [])
            # Prefer flash models for speed
            prefer = ["flash", "pro"]
            for keyword in prefer:
                for m in models:
                    methods = m.get("supportedGenerationMethods", [])
                    name    = m.get("name", "")          # e.g. "models/gemini-1.5-flash"
                    short   = name.replace("models/", "") # e.g. "gemini-1.5-flash"
                    if "generateContent" in methods and keyword in short:
                        return (api_ver, short)
            # Fallback — any model with generateContent
            for m in models:
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    short = m["name"].replace("models/", "")
                    return (api_ver, short)
        except Exception:
            continue
    return None


def gemini_ask(user_text: str) -> dict:
    """Send command to Gemini. Auto-discovers the correct model."""
    context = _build_ai_context()
    system  = AI_SYSTEM_PROMPT.replace("{context}", context)
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": system + "\n\nКоманда користувача: " + user_text}]}
        ],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
    }

    # ── Auto-discover model ──
    discovered = _get_available_gemini_model()
    if discovered is None:
        return {
            "action": "answer",
            "message": (
                "❌ Не вдалося отримати список моделей Gemini.\n"
                "Перевірте:\n"
                "• Чи правильний API ключ\n"
                "• Чи увімкнено Generative Language API у Google Cloud Console\n"
                "• Посилання: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com"
            )
        }

    api_ver, model = discovered
    url = (
        f"https://generativelanguage.googleapis.com/{api_ver}/models/"
        f"{model}:generateContent?key={GEMINI_API_KEY}"
    )

    try:
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except requests.exceptions.Timeout:
        return {"action": "answer", "message": "⏱️ Час очікування вийшов. Спробуйте ще раз."}
    except json.JSONDecodeError:
        return {"action": "answer", "message": "❌ AI повернув невалідну відповідь. Спробуйте ще раз."}
    except Exception as ex:
        return {"action": "answer", "message": f"❌ Помилка запиту: {ex}"}


def ai_execute(cmd: dict) -> tuple[str, str]:
    """
    Execute parsed AI command.
    Returns (status_emoji_message, level) where level in 'success'|'warning'|'error'|'info'
    """
    action  = cmd.get("action", "answer")
    message = cmd.get("message", "")

    if action == "log_glucose":
        level = float(cmd.get("level", 0))
        time_ = cmd.get("time", "Перед їжею")
        if level > 0:
            st.session_state.glucose_logs.append({
                "level": level, "time": time_,
                "timestamp": datetime.now().isoformat(),
                "date": datetime.now().strftime("%Y-%m-%d"),
            })
            save_data()
            lvl_type = "error" if (level < 4.0 or level >= 14.0) else "success"
            return message, lvl_type
        return "❌ Невірне значення глюкози", "error"

    elif action == "add_food":
        name    = cmd.get("name", "")
        carbs100 = float(cmd.get("carbs100", 0))
        weight  = float(cmd.get("weight", 100))
        protein = float(cmd.get("protein", 0))
        cal100  = float(cmd.get("calories100", 0))
        if add_product(name, carbs100, weight, protein, cal100):
            return message, "success"
        return "❌ Не вдалося додати продукт", "error"

    elif action == "calc_bolus":
        glucose = float(cmd.get("glucose", 5.5))
        ip      = st.session_state.insulin_profile
        p       = st.session_state.user_profile
        cr      = ip.get("cr", 10)
        isf     = ip.get("isf", 2.5)
        g_tgt   = p.get("target_min", 4.5)
        totals  = get_totals()
        carbs   = totals["carbs"]
        tiob    = total_iob(ip.get("active_doses", []), ip.get("iob_duration", 4.0))
        meal_d  = carbs / cr if cr > 0 else 0
        corr_d  = (glucose - g_tgt) / isf if isf > 0 else 0
        total_  = max(0.0, meal_d + corr_d - tiob)
        rounded = round(total_ * 2) / 2
        result  = (
            f"💉 Рекомендована доза: **{rounded:.1f} ОД**\n\n"
            f"• На їжу ({carbs:.0f} г): {meal_d:.2f} ОД\n"
            f"• Корекція (глюкоза {glucose}→{g_tgt}): {corr_d:+.2f} ОД\n"
            f"• Мінус IOB: −{tiob:.1f} ОД\n"
        )
        if glucose < 4.0:
            result += "\n🚨 **НЕБЕЗПЕКА**: гіпоглікемія — інсулін НЕ вводити!"
        return result, "warning" if glucose < 4.0 else "success"

    elif action == "log_dose":
        units = float(cmd.get("units", 0))
        note  = cmd.get("note", "")
        ip    = st.session_state.insulin_profile
        if units > 0:
            entry = {
                "units": units, "timestamp": datetime.now().isoformat(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "note": note, "manual": True,
                "glucose_before": None, "carbs": 0, "glucose_after": None,
            }
            st.session_state.dose_log.append(entry)
            ip.setdefault("active_doses", []).append({
                "units": units, "timestamp": datetime.now().isoformat()
            })
            save_data()
            return message, "success"
        return "❌ Невірна доза", "error"

    elif action == "log_ketones":
        value = float(cmd.get("value", 0))
        klabel, kcolor, kmsg = get_ketone_zone(value)
        st.session_state.ketone_logs.append({
            "value": value, "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"), "label": klabel,
        })
        save_data()
        lvl = "error" if value >= 3.1 else ("warning" if value >= 1.6 else "success")
        return f"{message}\n{klabel}: {kmsg}", lvl

    elif action == "save_meal":
        save_meal_snapshot()
        return message, "success"

    elif action == "clear_meal":
        st.session_state.meal_data = []
        save_data()
        return message, "info"

    else:  # answer
        return message, "info"


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
        "ketone_logs":     [],
        "dose_log":        [],
        "_loaded":         False,
        # UI state
        "app_theme":       "dark",
        "app_mode":        "lite",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state._loaded:
        d = load_data()
        if d:
            for key in ["meal_data","bu_weight","daily_totals","product_history",
                        "product_freq","glucose_logs","meal_patterns","meal_templates",
                        "ketone_logs","dose_log"]:
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
        return {"carbs": 0, "bu": 0, "cal": 0, "gl": 0, "protein": 0, "fat": 0}
    return {
        "carbs":   round(sum(i["Вугл."] for i in data), 1),
        "bu":      round(sum(i["ХО"]    for i in data), 2),
        "cal":     round(sum(i.get("Ккал", 0) for i in data), 0),
        "gl":      round(sum(i.get("ГН", 0)   for i in data), 1),
        "protein": round(sum(i.get("Білки", 0) for i in data), 1),
        "fat":     round(sum(i.get("Жири", 0) for i in data), 1),
    }


def add_product(name: str, carbs100: float, weight: float,
                protein: float = 0, calories_per100: float = 0,
                fat_per100: float = 0) -> bool:
    name = name.strip()
    if not name or len(name) < 2:
        st.error("Введіть назву продукту (мін. 2 символи)"); return False
    if not (0 <= carbs100 <= 100):
        st.error("Вуглеводи: 0–100 г на 100 г"); return False
    if not (1 <= weight <= 5000):
        st.error("Вага: 1–5000 г"); return False

    bu_w = st.session_state.bu_weight
    c    = calc_bu(carbs100, weight, bu_w)
    real_kcal   = round(calories_per100 * weight / 100, 0) if calories_per100 else c["calories"]
    protein_g   = round(protein * weight / 100, 1)
    fat_g       = round(fat_per100 * weight / 100, 1)

    st.session_state.meal_data.append({
        "Продукт": name,
        "Вага":    int(weight),
        "Вугл.":   c["carbs"],
        "ХО":      c["bu"],
        "Ккал":    real_kcal,
        "ГН":      c["gl"],
        "Білки":   protein_g,
        "Жири":    fat_g,
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
    t = get_totals()
    st.session_state.daily_totals.setdefault(today, []).append({
        "timestamp": datetime.now().isoformat(),
        "data":      st.session_state.meal_data.copy(),
        "totals":    t,
        "meal_type": get_meal_type(datetime.now().hour),
    })
    save_data()
    st.success("💾 Прийом збережено в журнал!")


# ══════════════════════════════════════════════════════════════════════════════
#  AI COMMAND BAR — одна стрічка для всього
# ══════════════════════════════════════════════════════════════════════════════

AI_EXAMPLES = [
    "Глюкоза 7.2 перед їжею",
    "Додай гречку 150г",
    "Скільки інсуліну при глюкозі 10.5?",
    "Записати 3.5 ОД інсуліну",
    "Кетони 0.4",
    "Зберегти прийом",
]


def render_ai_bar():
    C     = _theme_colors()
    theme = st.session_state.get("app_theme", "dark")

    bar_bg = "#0d1117" if theme == "dark" else "#ffffff"
    glow   = ("0 0 0 1px rgba(232,80,106,.35), 0 4px 24px rgba(232,80,106,.15)"
              if theme == "dark"
              else "0 0 0 1px rgba(232,80,106,.3), 0 4px 20px rgba(232,80,106,.1)")

    st.markdown(f"""
<style>
.ai-result-box {{
    background: {"rgba(255,255,255,.03)" if theme=="dark" else "rgba(0,0,0,.02)"};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 14px;
    line-height: 1.6;
    color: {C["text_primary"]};
}}
.ai-chip {{
    display: inline-block;
    background: {"rgba(232,80,106,.08)" if theme=="dark" else "rgba(232,80,106,.07)"};
    border: 1px solid rgba(232,80,106,.2);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 11px;
    color: #e8506a;
    margin: 2px;
    white-space: nowrap;
}}
</style>
""", unsafe_allow_html=True)

    if "ai_history" not in st.session_state:
        st.session_state.ai_history = []

    # Input row
    ai_col1, ai_col2 = st.columns([7, 1])
    with ai_col1:
        st.markdown(
            '<div style="font-size:11px;font-weight:700;letter-spacing:.08em;'
            'text-transform:uppercase;color:#e8506a;margin-bottom:6px">'
            '🤖 AI-асистент — введи будь-яку команду одним рядком</div>',
            unsafe_allow_html=True)
        user_cmd = st.text_input(
            "ai_input",
            placeholder='"Глюкоза 7.2"  ·  "Гречка 150г"  ·  "Скільки інсуліну при 10.5?"  ·  "Зберегти прийом"',
            key="ai_input_field",
            label_visibility="collapsed",
        )
    with ai_col2:
        st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
        send = st.button("⚡ Виконати", key="ai_send_btn", use_container_width=True)

    # Example chips
    chips = " ".join(f'<span class="ai-chip">↗ {e}</span>' for e in AI_EXAMPLES)
    st.markdown(f'<div style="margin:-4px 0 14px">{chips}</div>', unsafe_allow_html=True)

    # Process command
    if send and user_cmd.strip():
        with st.spinner("🤖 AI обробляє команду..."):
            cmd        = gemini_ask(user_cmd.strip())
            result_msg, level = ai_execute(cmd)

        st.session_state.ai_history.insert(0, {
            "user":   user_cmd.strip(),
            "result": result_msg,
            "level":  level,
            "time":   datetime.now().strftime("%H:%M"),
        })
        st.session_state.ai_history = st.session_state.ai_history[:10]
        st.rerun()

    # Show last 3 results
    if st.session_state.ai_history:
        level_colors = {"success": "#34d399", "warning": "#fbbf24",
                        "error": "#f87171", "info": "#60a5fa"}
        level_icons  = {"success": "✅", "warning": "⚠️",
                        "error": "🚨", "info": "💬"}
        for entry in st.session_state.ai_history[:3]:
            lc = level_colors.get(entry["level"], "#9ca3af")
            li = level_icons.get(entry["level"], "💬")
            result_html = entry["result"].replace("\n", "<br>")
            st.markdown(f"""
<div class="ai-result-box">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">
    <div style="font-size:12px;color:{C['text_faint']}">{li} <span style="color:{C['text_muted']}">{entry['user']}</span></div>
    <div style="font-size:10px;color:{C['text_faint']}">{entry['time']}</div>
  </div>
  <div style="color:{lc};font-size:13px;line-height:1.6">{result_html}</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider" style="margin:4px 0 20px"/>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  THEME CSS
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    theme = st.session_state.get("app_theme", "dark")

    if theme == "dark":
        C = {
            "bg_app":       "#060810",
            "bg_card":      "#0f1119",
            "bg_card2":     "#111318",
            "bg_input":     "#12141e",
            "bg_sidebar":   "linear-gradient(180deg,#0a0c14 0%,#080a12 100%)",
            "border":       "rgba(255,255,255,0.06)",
            "border_hover": "rgba(232,80,106,0.3)",
            "text_primary": "#eef0f5",
            "text_second":  "#d1d5db",
            "text_muted":   "#9ca3af",
            "text_faint":   "#4b5563",
            "tab_active_bg":"transparent",
            "metric_bg":    "#111318",
            "exp_bg":       "#111318",
            "tag_bg":       "#181c23",
            "iob_bg":       "#181c23",
            "dose_item_bg": "#12141e",
            "hover_label":  "#181c23",
            "hover_text":   "#eef0f5",
            "scrollbar":    "rgba(255,255,255,.08)",
            "divider":      "rgba(255,255,255,.05)",
            "grid":         "rgba(255,255,255,0.04)",
        }
    else:
        C = {
            "bg_app":       "#f0f4f8",
            "bg_card":      "#ffffff",
            "bg_card2":     "#f8fafc",
            "bg_input":     "#f1f5f9",
            "bg_sidebar":   "linear-gradient(180deg,#ffffff 0%,#f8fafc 100%)",
            "border":       "rgba(0,0,0,0.08)",
            "border_hover": "rgba(232,80,106,0.4)",
            "text_primary": "#1a1f36",
            "text_second":  "#374151",
            "text_muted":   "#6b7280",
            "text_faint":   "#9ca3af",
            "tab_active_bg":"transparent",
            "metric_bg":    "#ffffff",
            "exp_bg":       "#ffffff",
            "tag_bg":       "#e2e8f0",
            "iob_bg":       "#e2e8f0",
            "dose_item_bg": "#f8fafc",
            "hover_label":  "#1a1f36",
            "hover_text":   "#ffffff",
            "scrollbar":    "rgba(0,0,0,0.15)",
            "divider":      "rgba(0,0,0,0.07)",
            "grid":         "rgba(0,0,0,0.04)",
        }

    plotly_font   = C["text_muted"]
    plotly_grid   = C["grid"]
    plotly_hover_bg = C["hover_label"]
    plotly_hover_t  = C["hover_text"]
    plotly_paper_bg = "rgba(0,0,0,0)" if theme == "dark" else "rgba(255,255,255,0)"

    # Store plotly theme for use in chart functions
    st.session_state["_plotly_theme"] = dict(
        paper_bgcolor=plotly_paper_bg,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=plotly_font, family="'DM Mono', monospace", size=11),
        xaxis=dict(gridcolor=plotly_grid, zerolinecolor=plotly_grid, showline=False),
        yaxis=dict(gridcolor=plotly_grid, zerolinecolor=plotly_grid, showline=False),
        margin=dict(l=40, r=20, t=44, b=36),
        hoverlabel=dict(bgcolor=plotly_hover_bg, bordercolor=C["border"],
                        font=dict(color=plotly_hover_t, family="'Outfit', sans-serif")),
    )

    badge_target_bg   = "rgba(52,211,153,.1)"  if theme == "dark" else "rgba(52,211,153,.15)"
    badge_danger_bg   = "rgba(248,113,113,.12)" if theme == "dark" else "rgba(248,113,113,.12)"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Outfit', sans-serif !important; }}
.stApp {{ background: {C["bg_app"]} !important; color: {C["text_primary"]} !important; }}
.block-container {{ padding: 1.5rem 2rem 4rem !important; max-width: 1440px !important; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {C["bg_sidebar"]} !important;
    border-right: 1px solid {C["border"]} !important;
}}
[data-testid="stSidebar"] * {{ color: {C["text_second"]} !important; }}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] strong {{ color: {C["text_primary"]} !important; }}
section[data-testid="stSidebar"] > div {{ padding: 1.5rem 1rem !important; }}

/* ── Inputs ── */
.stTextInput input,.stNumberInput input,
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div:first-child {{
    background: {C["bg_input"]} !important;
    border: 1px solid {C["border"]} !important;
    border-radius: 10px !important;
    color: {C["text_primary"]} !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
    transition: border-color .2s, box-shadow .2s !important;
}}
.stTextInput input:focus,.stNumberInput input:focus {{
    border-color: rgba(232,80,106,.5) !important;
    box-shadow: 0 0 0 3px rgba(232,80,106,.07) !important;
}}
.stSelectbox label,.stNumberInput label,.stTextInput label {{
    font-size: 11px !important; color: {C["text_muted"]} !important;
    font-weight: 600 !important; letter-spacing: .06em !important;
    text-transform: uppercase !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg,#e8506a,#f97b4f) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-family: 'Outfit',sans-serif !important;
    font-weight: 600 !important; font-size: 13px !important;
    padding: 10px 20px !important; letter-spacing: .02em !important;
    box-shadow: 0 4px 16px rgba(232,80,106,.25) !important;
    transition: all .2s !important;
}}
.stButton > button:hover {{ transform: translateY(-1px) !important; box-shadow: 0 6px 24px rgba(232,80,106,.4) !important; }}
.stButton > button:active {{ transform: translateY(0) !important; }}
.stButton > button[kind="secondary"] {{
    background: {C["bg_input"]} !important; border: 1px solid {C["border"]} !important;
    box-shadow: none !important; color: {C["text_muted"]} !important;
}}
.stButton > button[kind="secondary"]:hover {{
    border-color: {C["border_hover"]} !important; color: {C["text_primary"]} !important;
    transform: none !important; box-shadow: none !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent !important; gap: 2px !important;
    border-bottom: 1px solid {C["border"]} !important; padding-bottom: 0 !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important; color: {C["text_faint"]} !important;
    font-family: 'Outfit',sans-serif !important; font-size: 13px !important;
    font-weight: 600 !important; padding: 10px 18px !important;
    border-radius: 8px 8px 0 0 !important; border: none !important;
    letter-spacing: .02em !important; transition: color .15s !important;
}}
.stTabs [aria-selected="true"] {{
    color: {C["text_primary"]} !important; background: transparent !important;
    border-bottom: 2px solid #e8506a !important;
}}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {{ color: {C["text_muted"]} !important; }}
.stTabs [data-baseweb="tab-panel"] {{ background: transparent !important; padding-top: 24px !important; }}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: {C["metric_bg"]} !important; border: 1px solid {C["border"]} !important;
    border-radius: 14px !important; padding: 16px 20px !important;
}}
[data-testid="stMetricValue"] {{ font-family: 'DM Mono',monospace !important; font-size: 1.4rem !important; color: {C["text_primary"]} !important; }}
[data-testid="stMetricLabel"] {{ font-size: 11px !important; color: {C["text_faint"]} !important; text-transform: uppercase !important; letter-spacing:.06em !important; }}
[data-testid="stMetricDelta"] {{ font-family: 'DM Mono',monospace !important; font-size: 12px !important; }}

/* ── Expander ── */
details > summary {{
    background: {C["exp_bg"]} !important; border: 1px solid {C["border"]} !important;
    border-radius: 10px !important; padding: 12px 16px !important;
    font-family: 'Outfit',sans-serif !important; font-size: 13px !important;
    font-weight: 600 !important; color: {C["text_second"]} !important;
    list-style: none !important; cursor: pointer !important;
    transition: border-color .2s !important;
}}
details > summary:hover {{ border-color: {C["border_hover"]} !important; }}
details[open] > summary {{ border-radius: 10px 10px 0 0 !important; border-bottom-color: transparent !important; }}
details > div {{
    background: {C["exp_bg"]} !important; border: 1px solid {C["border"]} !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important; padding: 12px 16px !important;
}}

/* ── DataFrame ── */
.stDataFrame > div {{ border-radius: 14px !important; overflow: hidden !important; }}
.stDataFrame [data-testid="stDataFrameResizable"] {{ background: {C["bg_card2"]} !important; }}

/* ── Alerts ── */
.stAlert {{ border-radius: 10px !important; font-family: 'Outfit',sans-serif !important; font-size: 13px !important; }}

/* ── Progress ── */
.stProgress > div > div {{ background: linear-gradient(90deg,#e8506a,#f97b4f) !important; border-radius: 9px !important; }}
.stProgress > div {{ background: {C["bg_input"]} !important; border-radius: 9px !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {C["scrollbar"]}; border-radius: 5px; }}

/* ── Custom HTML Components ── */
.glyco-title {{
    font-family: 'DM Serif Display',serif;
    background: linear-gradient(135deg,#e8506a,#f97b4f);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    line-height: 1.1; letter-spacing: -1px;
}}
.card {{
    background: {C["bg_card"]}; border: 1px solid {C["border"]};
    border-radius: 16px; padding: 20px 24px; margin-bottom: 16px;
}}
.card-sm {{ padding: 14px 16px; border-radius: 12px; }}
.card-accent {{
    background: linear-gradient(135deg,rgba(232,80,106,.07),rgba(249,123,79,.04));
    border-color: rgba(232,80,106,.18);
}}
.card-success {{ background: rgba(52,211,153,.06); border-color: rgba(52,211,153,.2); }}
.card-warning {{ background: rgba(251,191,36,.06); border-color: rgba(251,191,36,.2); }}
.card-danger  {{ background: rgba(248,113,113,.08); border-color: rgba(248,113,113,.3); }}

.result-hero {{ text-align: center; padding: 28px 24px; }}
.result-num {{
    font-family: 'DM Serif Display',serif; font-size: 4rem; line-height: 1;
    background: linear-gradient(135deg,#e8506a,#f97b4f);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    display: block;
}}
.result-unit {{ font-size: 13px; color: {C["text_muted"]}; margin-top: 4px; display: block; }}
.kpi-row {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin: 16px 0; }}
.kpi {{
    background: {'rgba(255,255,255,.03)' if theme=='dark' else 'rgba(0,0,0,.03)'};
    border-radius: 10px; padding: 12px 14px; text-align: center;
}}
.kpi-v {{ font-family: 'DM Mono',monospace; font-size: 1.15rem; font-weight: 500; display: block; }}
.kpi-l {{ font-size: 10px; color: {C["text_faint"]}; text-transform: uppercase; letter-spacing: .06em; margin-top: 2px; display: block; }}

.badge {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 14px; border-radius: 100px; font-size: 12px; font-weight: 600;
    border: 1px solid; margin: 2px;
}}
.badge-target   {{ background:{badge_target_bg};  color:#34d399; border-color:rgba(52,211,153,.25); }}
.badge-low      {{ background:rgba(251,146,60,.1);  color:#fb923c; border-color:rgba(251,146,60,.25); }}
.badge-danger   {{ background:{badge_danger_bg}; color:#f87171; border-color:rgba(248,113,113,.4);
                  animation: pulse-danger 1.5s ease-in-out infinite; }}
.badge-elevated {{ background:rgba(251,191,36,.1);  color:#fbbf24; border-color:rgba(251,191,36,.25); }}
.badge-info     {{ background:rgba(96,165,250,.1);  color:#60a5fa; border-color:rgba(96,165,250,.25); }}
.badge-high     {{ background:rgba(251,146,60,.1);  color:#fb923c; border-color:rgba(251,146,60,.25); }}
.badge-purple   {{ background:rgba(167,139,250,.1); color:#a78bfa; border-color:rgba(167,139,250,.25); }}
@keyframes pulse-danger {{
  0%,100% {{ box-shadow: 0 0 0 0 rgba(248,113,113,.3); }}
  50%      {{ box-shadow: 0 0 0 6px rgba(248,113,113,.0); }}
}}

.divider {{ border: none; border-top: 1px solid {C["divider"]}; margin: 20px 0; }}
.section-title {{ font-size: 11px; color: {C["text_faint"]}; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; margin-bottom: 12px; }}
.warn-bar {{
    display: flex; align-items: flex-start; gap: 8px;
    background: rgba(251,191,36,.06); border: 1px solid rgba(251,191,36,.2);
    border-radius: 10px; padding: 10px 14px; font-size: 12px; color: #fbbf24; line-height: 1.5;
}}
.tag {{
    display: inline-block; background: {C["tag_bg"]}; border: 1px solid {C["border"]};
    border-radius: 6px; padding: 3px 10px; font-size: 12px; color: {C["text_muted"]}; margin: 2px;
    cursor: pointer; transition: all .15s;
}}
.tag:hover {{ border-color: rgba(232,80,106,.4); color: {C["text_primary"]}; }}
.iob-bar {{ background: {C["iob_bg"]}; border-radius: 10px; padding: 12px 14px; }}
.tir-legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }}
.tir-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: {C["text_muted"]}; }}
.tir-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
.streak-badge {{
    display: flex; align-items: center; gap: 8px;
    background: linear-gradient(135deg,rgba(251,191,36,.12),rgba(249,123,79,.08));
    border: 1px solid rgba(251,191,36,.25); border-radius: 12px;
    padding: 10px 14px; margin-top: 8px;
}}
.insight-card {{
    background: rgba(96,165,250,.05); border: 1px solid rgba(96,165,250,.15);
    border-radius: 12px; padding: 14px 16px; margin-bottom: 10px;
}}
.dose-item {{
    display: flex; justify-content: space-between; align-items: center;
    background: {C["dose_item_bg"]}; border: 1px solid {C["border"]};
    border-radius: 10px; padding: 10px 14px; margin-bottom: 6px;
}}
.trend-arrow {{ font-size: 1.4rem; font-weight: bold; }}
.trend-up2   {{ color: #f87171; }}
.trend-up    {{ color: #fbbf24; }}
.trend-flat  {{ color: #34d399; }}
.trend-down  {{ color: #60a5fa; }}
.trend-down2 {{ color: #818cf8; }}

/* ── Mode toggle button style ── */
.mode-btn {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 16px; border-radius: 20px; font-size: 12px; font-weight: 600;
    border: 1px solid {C["border"]}; cursor: pointer;
    transition: all .2s; color: {C["text_muted"]}; background: {C["bg_input"]};
}}
.mode-btn.active {{
    background: linear-gradient(135deg,#e8506a,#f97b4f);
    border-color: transparent; color: #fff;
    box-shadow: 0 4px 12px rgba(232,80,106,.3);
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _theme_colors():
    theme = st.session_state.get("app_theme", "dark")
    if theme == "dark":
        return {
            "text_muted":   "#9ca3af",
            "text_faint":   "#4b5563",
            "text_second":  "#d1d5db",
            "text_primary": "#eef0f5",
            "bg_card":      "#0f1119",
            "bg_input":     "#12141e",
            "border":       "rgba(255,255,255,0.06)",
        }
    else:
        return {
            "text_muted":   "#6b7280",
            "text_faint":   "#9ca3af",
            "text_second":  "#374151",
            "text_primary": "#1a1f36",
            "bg_card":      "#ffffff",
            "bg_input":     "#f1f5f9",
            "border":       "rgba(0,0,0,0.08)",
        }


def trend_arrow_html(arrow: str) -> str:
    cls_map = {"↑↑": "trend-up2", "↑": "trend-up", "→": "trend-flat",
               "↓": "trend-down", "↓↓": "trend-down2"}
    label_map = {"↑↑": "Швидко ↑↑", "↑": "Зростає ↑", "→": "Стабільно →",
                 "↓": "Знижується ↓", "↓↓": "Швидко ↓↓"}
    cls   = cls_map.get(arrow, "trend-flat")
    label = label_map.get(arrow, arrow)
    return f'<span class="trend-arrow {cls}" title="Тренд глюкози">{arrow}</span> <span style="font-size:11px;color:#6b7280">{label}</span>'


def glucose_ring_html(value: Optional[float], trend: str = "→") -> str:
    C = _theme_colors()
    if value is None:
        return f"""
        <div style="display:flex;flex-direction:column;align-items:center;padding:16px 0 8px">
          <svg width="150" height="150" viewBox="0 0 150 150">
            <circle cx="75" cy="75" r="62" fill="none" stroke="{C['bg_input']}" stroke-width="10"/>
          </svg>
          <div style="font-size:12px;color:{C['text_faint']};margin-top:-70px;padding-bottom:50px">введіть значення</div>
        </div>"""
    label, key, color, msg = get_zone(value)
    pct  = min(1.0, max(0.0, (value - 2.0) / 18.0))
    circ = 2 * math.pi * 62
    dash = circ * pct
    glow = f"filter:drop-shadow(0 0 8px {color}88);" if key == "danger" else ""
    trend_cls_map = {"↑↑": "trend-up2", "↑": "trend-up", "→": "trend-flat",
                     "↓": "trend-down", "↓↓": "trend-down2"}
    trend_cls = trend_cls_map.get(trend, "trend-flat")
    track_color = "#1a1d24" if st.session_state.get("app_theme","dark") == "dark" else "#e2e8f0"
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;padding:12px 0 4px">
      <div style="position:relative;width:150px;height:150px">
        <svg width="150" height="150" viewBox="0 0 150 150" style="transform:rotate(-90deg);{glow}">
          <circle cx="75" cy="75" r="62" fill="none" stroke="{track_color}" stroke-width="10"/>
          <circle cx="75" cy="75" r="62" fill="none" stroke="{color}" stroke-width="10"
            stroke-linecap="round" stroke-dasharray="{dash:.1f} {circ:.1f}"
            style="transition:all 0.8s cubic-bezier(.4,0,.2,1)"/>
        </svg>
        <div style="position:absolute;inset:0;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;gap:1px">
          <span style="font-family:'DM Serif Display',serif;font-size:2.1rem;
                        color:{color};line-height:1">{value}</span>
          <span style="font-size:10px;color:{C['text_muted']}">ммоль/л</span>
          <span class="trend-arrow {trend_cls}" style="font-size:1rem;margin-top:2px">{trend}</span>
        </div>
      </div>
      <div style="margin-top:8px">
        <span class="badge badge-{key}">{label}</span>
      </div>
      <div style="font-size:11px;color:{C['text_muted']};text-align:center;max-width:160px;
                  line-height:1.4;margin-top:6px;min-height:30px">{msg}</div>
    </div>"""


def mini_card(title: str, value: str, color: str = "#e8506a", sub: str = "") -> str:
    C = _theme_colors()
    return f"""
    <div class="card card-sm" style="text-align:center">
      <div style="font-size:10px;color:{C['text_faint']};text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:6px">{title}</div>
      <div style="font-family:'DM Mono',monospace;font-size:1.4rem;
                  font-weight:500;color:{color}">{value}</div>
      {f'<div style="font-size:11px;color:{C["text_muted"]};margin-top:3px">{sub}</div>' if sub else ''}
    </div>"""


def result_card(bu: float, carbs: float, cal: float, gl: float,
                protein: float = 0, fat: float = 0) -> str:
    fat_block = f"""
        <div class="kpi">
          <span class="kpi-v" style="color:#a78bfa">{protein} г</span>
          <span class="kpi-l">Білки</span>
        </div>
        <div class="kpi">
          <span class="kpi-v" style="color:#818cf8">{fat} г</span>
          <span class="kpi-l">Жири</span>
        </div>""" if (protein or fat) else ""
    cols = "repeat(3,1fr)" if not (protein or fat) else "repeat(5,1fr)"
    return f"""
    <div class="card card-accent result-hero">
      <span style="font-size:11px;color:#6b7280;text-transform:uppercase;
                   letter-spacing:.1em;margin-bottom:8px;display:block">Разом за прийом</span>
      <span class="result-num">{bu}</span>
      <span class="result-unit">хлібних одиниць</span>
      <div class="kpi-row" style="margin-top:20px;grid-template-columns:{cols}">
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
        {fat_block}
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


def hba1c_card(hba1c: float, gmi: float, avg_g: float, tir_pct: float, cv: float) -> str:
    hba1c_color = "#34d399" if hba1c <= 7.0 else ("#fbbf24" if hba1c <= 8.5 else "#f87171")
    hba1c_label = "Відмінно" if hba1c <= 7.0 else ("Добре" if hba1c <= 8.0 else "Потребує уваги")
    cv_color    = "#34d399" if cv < 36 else ("#fbbf24" if cv < 46 else "#f87171")
    cv_label    = "Стабільно" if cv < 36 else ("Помірно" if cv < 46 else "Варіабельно")
    C = _theme_colors()
    sep = f"border-left:1px solid {C['border']}"
    return f"""
    <div class="card" style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;gap:16px;padding:20px 24px">
      <div style="text-align:center">
        <div style="font-size:10px;color:{C['text_faint']};text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">HbA1c (розрахунк.)</div>
        <div style="font-family:'DM Serif Display',serif;font-size:2.2rem;color:{hba1c_color};line-height:1">{hba1c}%</div>
        <div style="font-size:11px;color:{hba1c_color};margin-top:4px">{hba1c_label}</div>
      </div>
      <div style="text-align:center;{sep}">
        <div style="font-size:10px;color:{C['text_faint']};text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">GMI (точніше)</div>
        <div style="font-family:'DM Mono',monospace;font-size:2rem;color:#a78bfa;line-height:1">{gmi}%</div>
        <div style="font-size:11px;color:{C['text_muted']};margin-top:4px">ціль: &lt;7%</div>
      </div>
      <div style="text-align:center;{sep}">
        <div style="font-size:10px;color:{C['text_faint']};text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Середня глюкоза</div>
        <div style="font-family:'DM Mono',monospace;font-size:2rem;color:#60a5fa;line-height:1">{avg_g}</div>
        <div style="font-size:11px;color:{C['text_muted']};margin-top:4px">ммоль/л</div>
      </div>
      <div style="text-align:center;{sep}">
        <div style="font-size:10px;color:{C['text_faint']};text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Час у нормі (TIR)</div>
        <div style="font-family:'DM Mono',monospace;font-size:2rem;color:#34d399;line-height:1">{tir_pct}%</div>
        <div style="font-size:11px;color:{C['text_muted']};margin-top:4px">ціль: ≥ 70%</div>
      </div>
      <div style="text-align:center;{sep}">
        <div style="font-size:10px;color:{C['text_faint']};text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">CV% варіабельність</div>
        <div style="font-family:'DM Mono',monospace;font-size:2rem;color:{cv_color};line-height:1">{cv}%</div>
        <div style="font-size:11px;color:{cv_color};margin-top:4px">{cv_label} · ціль &lt;36%</div>
      </div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        # ── Logo + Mode/Theme row ──
        st.markdown('<div class="glyco-title" style="font-size:1.7rem;margin-bottom:6px">GlyPro</div>', unsafe_allow_html=True)

        # Mode toggle
        mode = st.session_state.app_mode
        theme = st.session_state.app_theme

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            lite_style = "background:linear-gradient(135deg,#e8506a,#f97b4f);color:#fff;border-color:transparent;" if mode == "lite" else ""
            if st.button("⚡ Lite", use_container_width=True, key="btn_mode_lite"):
                st.session_state.app_mode = "lite"
                st.rerun()
        with col_m2:
            if st.button("🔬 Pro", use_container_width=True, key="btn_mode_pro"):
                st.session_state.app_mode = "pro"
                st.rerun()

        # Show mode indicator
        mode_label = "⚡ Lite — спрощений" if mode == "lite" else "🔬 Pro — повний"
        st.markdown(f'<div style="font-size:10px;text-align:center;color:#6b7280;margin-bottom:4px">{mode_label}</div>', unsafe_allow_html=True)

        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

        # Theme toggle
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if st.button("🌙 Темна", use_container_width=True, key="btn_theme_dark"):
                st.session_state.app_theme = "dark"
                st.rerun()
        with col_t2:
            if st.button("☀️ Світла", use_container_width=True, key="btn_theme_light"):
                st.session_state.app_theme = "light"
                st.rerun()

        theme_label = "🌙 Темна тема" if theme == "dark" else "☀️ Світла тема"
        st.markdown(f'<div style="font-size:10px;text-align:center;color:#6b7280;margin-bottom:4px">{theme_label}</div>', unsafe_allow_html=True)

        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

        # ── Live Glucose ──
        st.markdown('<div class="section-title">🩸 Поточна глюкоза</div>', unsafe_allow_html=True)
        g_now = st.number_input("Глюкоза (ммоль/л)", 0.5, 35.0, step=0.1,
                                 key="sidebar_glucose", label_visibility="collapsed")
        glogs = st.session_state.glucose_logs
        trend = glucose_trend_arrow(glogs)

        if g_now and g_now > 0:
            st.markdown(glucose_ring_html(g_now, trend), unsafe_allow_html=True)
        else:
            st.markdown(glucose_ring_html(None), unsafe_allow_html=True)

        g_time = st.selectbox("Момент вимірювання",
            ["Перед їжею","Після їжі (1 год)","Після їжі (2 год)","Вранці натще","Перед сном"],
            key="sidebar_g_time", label_visibility="collapsed")

        if st.button("📝 Записати глюкозу", use_container_width=True, key="sb_log_g"):
            if g_now > 0:
                glogs.append({
                    "level": g_now, "time": g_time,
                    "timestamp": datetime.now().isoformat(),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                })
                save_data()
                st.success("✅ Записано!")
                st.rerun()

        if glogs:
            recent = glogs[-4:]
            st.markdown('<div class="section-title" style="margin-top:12px">Останні вимірювання</div>', unsafe_allow_html=True)
            for e in reversed(recent):
                lv = e["level"]
                _, key, color, _ = get_zone(lv)
                badge_cls = f"badge-{key}" if key != "unknown" else "badge-info"
                ts = e["timestamp"][11:16] if "T" in e.get("timestamp","") else e.get("time","")
                st.markdown(
                    f'<span class="badge {badge_cls}" style="font-size:11px">{lv} мм</span>'
                    f'<span style="font-size:10px;color:#4b5563;margin-left:6px">{ts}</span>',
                    unsafe_allow_html=True)

        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

        # Ketones (Pro only)
        if mode == "pro":
            st.markdown('<div class="section-title">🧪 Кетони (ммоль/л)</div>', unsafe_allow_html=True)
            k_val = st.number_input("Кетони", 0.0, 10.0, step=0.1,
                                     key="sb_ketone", label_visibility="collapsed", format="%.1f")
            if st.button("📝 Записати кетони", use_container_width=True, key="sb_log_k"):
                if k_val >= 0:
                    klabel, kcolor, kmsg = get_ketone_zone(k_val)
                    st.session_state.ketone_logs.append({
                        "value": k_val, "timestamp": datetime.now().isoformat(),
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "label": klabel,
                    })
                    save_data()
                    st.markdown(f'<span style="color:{kcolor};font-size:12px">{klabel}: {kmsg}</span>', unsafe_allow_html=True)
                    if k_val >= 3.1:
                        st.error("🚨 НЕБЕЗПЕЧНИЙ рівень кетонів! Негайно до лікарні!")
                    elif k_val >= 1.6:
                        st.warning("⚠️ Підвищені кетони. Зверніться до лікаря!")
                    st.rerun()

            if st.session_state.ketone_logs:
                last_k = st.session_state.ketone_logs[-1]
                klabel, kcolor, _ = get_ketone_zone(last_k["value"])
                st.markdown(
                    f'<span style="font-size:11px;color:#4b5563">Останнє: </span>'
                    f'<span style="color:{kcolor};font-size:12px;font-weight:600">{last_k["value"]} ммоль/л {klabel}</span>',
                    unsafe_allow_html=True)

            st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

        # ── Streak (Pro) ──
        if mode == "pro":
            profile = st.session_state.user_profile
            streak  = calc_streak(glogs, profile.get("target_min", 4.5), profile.get("target_max", 7.8))
            if streak > 0:
                streak_emoji = "🔥" * min(streak, 5)
                st.markdown(f"""
                <div class="streak-badge">
                  <div style="font-size:1.5rem">{streak_emoji}</div>
                  <div>
                    <div style="font-weight:700;color:#fbbf24;font-size:14px">{streak} {'день' if streak == 1 else ('дні' if streak < 5 else 'днів')} у нормі!</div>
                    <div style="font-size:11px;color:#6b7280">TIR ≥ 70% — чудово!</div>
                  </div>
                </div>""", unsafe_allow_html=True)
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
        st.caption(f"1 ХО = {bu} г")

        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

        # ── Quick Meal Actions ──
        if st.session_state.meal_data:
            t = get_totals()
            C = _theme_colors()
            st.markdown(f"""
            <div class="card card-sm" style="margin-bottom:12px">
              <div class="section-title" style="margin-bottom:8px">📊 Поточний прийом</div>
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                  <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;
                       background:linear-gradient(135deg,#e8506a,#f97b4f);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                       background-clip:text;line-height:1">{t["bu"]} ХО</div>
                  <div style="font-size:11px;color:{C['text_muted']}">{t["carbs"]} г · {int(t["cal"])} ккал</div>
                </div>
                <div style="font-size:1.5rem">🍽️</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        sb1, sb2 = st.columns(2)
        with sb1:
            if st.button("💾 Зберегти", use_container_width=True, key="sb_save"):
                save_meal_snapshot()
        with sb2:
            if st.button("🗑️ Очистити", use_container_width=True, key="sb_clear"):
                st.session_state.meal_data = []
                save_data()
                st.rerun()

        # ── IOB bar ──
        ip = st.session_state.insulin_profile
        if ip.get("active_doses"):
            tiob = total_iob(ip["active_doses"], ip["iob_duration"])
            if tiob > 0:
                st.markdown(f"""
                <div class="iob-bar" style="margin-top:8px">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                    <span style="font-size:11px;color:#6b7280">⏳ Активний IOB</span>
                    <span style="font-family:'DM Mono',monospace;color:#60a5fa;font-size:13px">{tiob} ОД</span>
                  </div>
                  <div style="background:rgba(255,255,255,.05);border-radius:6px;height:4px;overflow:hidden">
                    <div style="background:linear-gradient(90deg,#60a5fa,#e8506a);width:100%;height:100%;border-radius:6px"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

        # ── AI Status check ──
        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
        if st.button("🔌 Перевірити AI (Gemini)", use_container_width=True, key="btn_api_check"):
            with st.spinner("Запитую список моделей..."):
                found_lines = []
                for api_ver in ("v1beta", "v1"):
                    url = (
                        f"https://generativelanguage.googleapis.com/{api_ver}/models"
                        f"?key={GEMINI_API_KEY}"
                    )
                    try:
                        r = requests.get(url, timeout=10)
                        if r.status_code == 200:
                            models = r.json().get("models", [])
                            gc = [m["name"] for m in models
                                  if "generateContent" in m.get("supportedGenerationMethods", [])]
                            for name in gc:
                                found_lines.append(f'<span style="font-size:11px;color:#34d399">✅ {api_ver} / {name.replace("models/","")}</span>')
                            break
                        elif r.status_code == 403:
                            found_lines.append('<span style="font-size:11px;color:#f87171">🔑 API ключ недійсний або немає дозволів (403)</span>')
                            break
                        else:
                            found_lines.append(f'<span style="font-size:11px;color:#fbbf24">⚠️ {api_ver}: HTTP {r.status_code}</span>')
                    except Exception as e:
                        found_lines.append(f'<span style="font-size:11px;color:#f87171">❌ {api_ver}: {e}</span>')

            if found_lines:
                st.markdown("<br>".join(found_lines), unsafe_allow_html=True)
            else:
                st.error("Не вдалося отримати відповідь від Google API")


# ══════════════════════════════════════════════════════════════════════════════
#  LITE MODE — спрощений інтерфейс
# ══════════════════════════════════════════════════════════════════════════════

def render_lite():
    C = _theme_colors()
    # Header
    st.markdown(
        '<div class="glyco-title" style="font-size:2.2rem">GlyPro <span style="font-size:1rem;opacity:.5">Lite</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:12px;color:{C["text_faint"]};margin-bottom:16px">'
        f'Щоденник діабетика · {datetime.now().strftime("%d.%m.%Y %H:%M")}</div>',
        unsafe_allow_html=True)

    # ── AI BAR ──
    render_ai_bar()

    # ── Section 1: Їжа ──
    st.markdown(f'<div style="font-size:13px;font-weight:700;color:{C["text_second"]};margin-bottom:12px;text-transform:uppercase;letter-spacing:.08em">🍽️ Додати їжу</div>', unsafe_allow_html=True)

    db = load_product_db()

    # Two sub-tabs: DB search vs Manual
    lt1, lt2 = st.tabs(["🔍 З бази", "✏️ Вручну"])

    with lt1:
        if db:
            sc1, sc2, sc3 = st.columns([2.5, 1, 1])
            with sc1:
                search_q = st.text_input("Пошук", placeholder="хліб, гречка, яблуко...",
                                          key="lite_db_search", label_visibility="collapsed")
            with sc2:
                qa_wt = st.number_input("Порція (г)", 1, 2000, 100, step=10,
                                         key="lite_qa_wt", label_visibility="collapsed")
            with sc3:
                cat = st.selectbox("Категорія", ["(всі)"] + list(db.keys()), key="lite_qa_cat",
                                    label_visibility="collapsed")

            if search_q:
                q = search_q.lower()
                all_items = [item for cat_items in db.values() for item in cat_items]
                items = [i for i in all_items if q in i["name"].lower()][:12]
            elif cat != "(всі)":
                items = db.get(cat, [])[:12]
            else:
                items = []

            if items:
                sel = st.selectbox("Оберіть продукт",
                                    [p["name"] for p in items],
                                    key="lite_qa_sel", label_visibility="collapsed")
                p_info = next(p for p in items if p["name"] == sel)
                bu_preview = round((p_info["carbs"] * qa_wt / 100) / st.session_state.bu_weight, 2)
                st.markdown(
                    f'<span class="badge badge-info">{p_info["carbs"]} г вугл./100г</span>'
                    f'<span class="badge badge-target">≈ {bu_preview} ХО / {qa_wt}г</span>',
                    unsafe_allow_html=True)
                if st.button("➕ Додати до прийому", key="lite_btn_qa", use_container_width=True):
                    if add_product(sel, p_info["carbs"], qa_wt, p_info["protein"], p_info["calories"]):
                        st.rerun()
            elif search_q:
                st.info(f"«{search_q}» — не знайдено")
        else:
            st.info("База продуктів не завантажена (table.csv)")

    with lt2:
        mc1, mc2, mc3, mc4 = st.columns([3, 1.5, 1.5, 1])
        with mc1:
            p_name = st.text_input("Продукт", placeholder="Гречка варена", key="lite_m_name",
                                    label_visibility="collapsed")
        with mc2:
            p_c100 = st.number_input("Вугл./100г", 0.0, 100.0, step=0.5,
                                      format="%.1f", key="lite_m_carbs", label_visibility="collapsed")
        with mc3:
            p_wt = st.number_input("Вага (г)", 1, 5000, 100, step=5,
                                    key="lite_m_weight", label_visibility="collapsed")
        with mc4:
            st.markdown("<br/>", unsafe_allow_html=True)
            if st.button("➕", key="lite_btn_add_manual", use_container_width=True):
                if add_product(p_name, p_c100, p_wt):
                    st.rerun()

    # Frequent quick-add tags
    freq = st.session_state.product_freq
    if freq:
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:6]
        tags_html = "".join(f'<span class="tag">↩ {name}</span>' for name, _ in top)
        st.markdown(f'<div style="margin:8px 0 0">{tags_html}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    # ── Current meal table ──
    if st.session_state.meal_data:
        df = pd.DataFrame(st.session_state.meal_data)
        cols_show = [c for c in ["Продукт","Вага","Вугл.","ХО","Ккал"] if c in df.columns]
        st.dataframe(
            df[cols_show],
            use_container_width=True, hide_index=True,
            column_config={
                "Продукт": st.column_config.TextColumn(width="large"),
                "Вага":    st.column_config.NumberColumn("г", format="%d"),
                "Вугл.":   st.column_config.NumberColumn("Вугл. г", format="%.1f"),
                "ХО":      st.column_config.NumberColumn("ХО", format="%.2f ⭐"),
                "Ккал":    st.column_config.NumberColumn("ккал", format="%.0f"),
            },
            height=min(200, 40 + 35 * len(st.session_state.meal_data)),
            key="lite_meal_df"
        )

        if len(st.session_state.meal_data) > 1:
            rc1, rc2 = st.columns([4, 1])
            with rc1:
                to_rm = st.selectbox("Видалити", range(len(st.session_state.meal_data)),
                    format_func=lambda i: f"{st.session_state.meal_data[i]['Продукт']} ({st.session_state.meal_data[i]['ХО']} ХО)",
                    key="lite_rm_sel", label_visibility="collapsed")
            with rc2:
                if st.button("🗑️", key="lite_btn_rm", use_container_width=True):
                    st.session_state.meal_data.pop(to_rm)
                    save_data(); st.rerun()

        t = get_totals()
        st.markdown(result_card(t["bu"], t["carbs"], t["cal"], t["gl"], t["protein"], t["fat"]),
                    unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:32px;color:{C['text_faint']};font-style:italic">
            🍽️ Прийом порожній. Додайте продукти вище.
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    # ── Section 2: Інсулін ──
    st.markdown(f'<div style="font-size:13px;font-weight:700;color:{C["text_second"]};margin-bottom:12px;text-transform:uppercase;letter-spacing:.08em">💉 Болюс калькулятор</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-bar">⚠️ Тільки для ознайомлення. Дози погоджуйте з ендокринологом!</div>
    """, unsafe_allow_html=True)

    ip = st.session_state.insulin_profile

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        cr  = st.number_input("🍞 CR (г/ОД)", 5, 30, int(ip["cr"]), step=1, key="lite_ins_cr")
    with ic2:
        isf = st.number_input("📉 ISF (мм/ОД)", 0.5, 8.0, float(ip["isf"]), step=0.1, key="lite_ins_isf")
    with ic3:
        iob_dur = st.number_input("⏳ IOB тривалість (год)", 2.0, 8.0, float(ip["iob_duration"]), step=0.5, key="lite_ins_dur")

    if cr != ip["cr"] or isf != ip["isf"] or iob_dur != ip["iob_duration"]:
        ip.update({"cr": cr, "isf": isf, "iob_duration": iob_dur})
        save_data()

    gc1, gc2 = st.columns(2)
    with gc1:
        g_cur = st.number_input("🩸 Поточна глюкоза (ммоль/л)", 0.5, 35.0, 5.5, step=0.1, key="lite_ins_gcur")
    with gc2:
        g_tgt = st.number_input("🎯 Ціль (ммоль/л)", 3.0, 12.0,
                                  float(st.session_state.user_profile["target_min"]),
                                  step=0.1, key="lite_ins_gtgt")

    label, key_z, color, msg = get_zone(g_cur)
    st.markdown(f'<span class="badge badge-{key_z}">{label}</span> <span style="font-size:12px;color:{C["text_muted"]}">{msg}</span>', unsafe_allow_html=True)

    # Auto TOD
    auto_tod = auto_tod_key()
    tod_key  = st.radio("Час доби", list(TOD_ISF_FACTORS.keys()),
                         index=list(TOD_ISF_FACTORS.keys()).index(auto_tod),
                         horizontal=True, key="lite_ins_tod", label_visibility="collapsed")
    tod_factor = TOD_ISF_FACTORS[tod_key]
    adj_isf    = isf * tod_factor

    # IOB
    active_doses = ip.get("active_doses", [])
    cumulative_iob = total_iob(active_doses, iob_dur)
    if cumulative_iob > 0:
        st.markdown(f'<span class="badge badge-info">⏳ IOB: {cumulative_iob} ОД</span>', unsafe_allow_html=True)

    if st.button("⚡ Розрахувати дозу", use_container_width=True, key="lite_ins_calc"):
        totals   = get_totals()
        carbs    = totals["carbs"]
        meal_dose = carbs / cr if cr > 0 else 0
        corr_dose = (g_cur - g_tgt) / adj_isf if adj_isf > 0 else 0
        total_    = max(0.0, meal_dose + corr_dose - cumulative_iob)
        rounded   = round(total_ * 2) / 2

        st.markdown(dose_card(total_, rounded, meal_dose, corr_dose, cumulative_iob, carbs, tod_factor),
                    unsafe_allow_html=True)

        if g_cur < 4.0:
            st.error("🚨 ГІПОГЛІКЕМІЯ! НЕ вводьте інсулін! З'їжте 15 г швидких вуглеводів.")
        elif g_cur >= 14.0:
            st.warning("⚠️ Дуже високий рівень. Перевірте кетони.")

        if st.button(f"💉 Записати {rounded:.1f} ОД", key="lite_log_dose_btn"):
            new_dose = {
                "units": rounded, "timestamp": datetime.now().isoformat(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "carbs": carbs, "glucose_before": g_cur,
                "note": f"{get_meal_type(datetime.now().hour)}",
            }
            ip.setdefault("active_doses", []).append(new_dose)
            st.session_state.dose_log.append({
                **new_dose, "glucose_after": None,
            })
            cutoff = datetime.now() - timedelta(hours=iob_dur + 1)
            ip["active_doses"] = [d for d in ip["active_doses"]
                                   if datetime.fromisoformat(d["timestamp"]) > cutoff]
            save_data()
            st.success(f"✅ {rounded:.1f} ОД записано!")
            st.rerun()

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    # ── Quick Stats ──
    glogs = st.session_state.glucose_logs
    if glogs:
        stats = tir_stats(glogs[-30:] if len(glogs) > 30 else glogs)
        st.markdown(f'<div style="font-size:13px;font-weight:700;color:{C["text_second"]};margin-bottom:12px;text-transform:uppercase;letter-spacing:.08em">📊 Короткий звіт</div>', unsafe_allow_html=True)
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("TIR", f"{stats['target']}%", delta="ціль ≥70%")
        qc2.metric("Середня", f"{stats['avg']} мм")
        qc3.metric("HbA1c ~", f"{stats['hba1c']}%")
        qc4.metric("Вимірювань", stats['n'])


# ══════════════════════════════════════════════════════════════════════════════
#  PRO MODE TABS
# ══════════════════════════════════════════════════════════════════════════════

def tab_meal():
    C = _theme_colors()
    st.markdown('<div class="section-title">✏️ Додати вручну</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([2.5, 1.6, 1.6, 1])
    with c1:
        p_name = st.text_input("Продукт", placeholder="Гречка варена", key="m_name",
                                label_visibility="collapsed")
    with c2:
        p_c100 = st.number_input("Вугл./100г", 0.0, 100.0, step=0.5,
                                  format="%.1f", key="m_carbs", label_visibility="collapsed")
    with c3:
        p_wt = st.number_input("Вага (г)", 1, 5000, 100, step=5,
                                key="m_weight", label_visibility="collapsed")
    with c4:
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("➕ Додати", key="btn_add_manual", use_container_width=True):
            if add_product(p_name, p_c100, p_wt):
                st.rerun()

    freq = st.session_state.product_freq
    if freq:
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8]
        tags_html = "".join(f'<span class="tag">↩ {name}</span>' for name, _ in top)
        st.markdown(
            f'<div style="margin:8px 0 0"><div class="section-title">Часті продукти</div>'
            f'{tags_html}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">⚡ База продуктів (CSV)</div>', unsafe_allow_html=True)
    db = load_product_db()

    if db:
        cc1, cc2, cc3 = st.columns([2, 1, 1.5])
        with cc1:
            cat = st.selectbox("Категорія", list(db.keys()), key="qa_cat",
                                label_visibility="collapsed")
        with cc2:
            qa_wt = st.number_input("Порція (г)", 1, 2000, 100, step=10,
                                     key="qa_wt", label_visibility="collapsed")
        with cc3:
            search_q = st.text_input("🔍 Пошук продукту", placeholder="хліб, рис...",
                                     key="db_search", label_visibility="collapsed")

        items = db.get(cat, [])
        if search_q:
            q = search_q.lower()
            all_items = [item for cat_items in db.values() for item in cat_items]
            items = [i for i in all_items if q in i["name"].lower()]

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
        elif search_q:
            st.info(f"Продукт '{search_q}' не знайдено в базі")

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

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
                        save_data(); st.rerun()
                with tc3:
                    if st.button("🗑️", key=f"del_t_{name}"):
                        del templates[name]
                        save_data(); st.rerun()

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    if not st.session_state.meal_data:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:48px;color:{C['text_faint']};font-style:italic">
            🍽️ Прийом порожній. Додайте продукти вище.
        </div>""", unsafe_allow_html=True)
        return

    st.markdown('<div class="section-title">🍽️ Склад прийому</div>', unsafe_allow_html=True)
    df = pd.DataFrame(st.session_state.meal_data)

    cols_show = [c for c in ["Продукт","Вага","Вугл.","ХО","Ккал","Білки","Жири","ГН","Час"] if c in df.columns]
    st.dataframe(
        df[cols_show],
        use_container_width=True, hide_index=True,
        column_config={
            "Продукт": st.column_config.TextColumn(width="large"),
            "Вага":    st.column_config.NumberColumn("Вага (г)", format="%d г"),
            "Вугл.":   st.column_config.NumberColumn("Вуглеводи", format="%.1f г"),
            "ХО":      st.column_config.NumberColumn("ХО", format="%.2f ⭐"),
            "Ккал":    st.column_config.NumberColumn("Ккал", format="%.0f"),
            "Білки":   st.column_config.NumberColumn("Білки г", format="%.1f"),
            "Жири":    st.column_config.NumberColumn("Жири г", format="%.1f"),
            "ГН":      st.column_config.NumberColumn("Глік. навант.", format="%.1f"),
        },
        key="meal_df"
    )

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

    t = get_totals()
    st.markdown(result_card(t["bu"], t["carbs"], t["cal"], t["gl"], t["protein"], t["fat"]),
                unsafe_allow_html=True)


def tab_insulin():
    st.markdown("""
    <div class="warn-bar">
      ⚠️ Тільки для ознайомлення. Будь-яке коригування доз — виключно з ендокринологом!
    </div>""", unsafe_allow_html=True)

    ip = st.session_state.insulin_profile
    PLOTLY_THEME = st.session_state.get("_plotly_theme", {})

    st.markdown('<div class="section-title">📐 Параметри інсуліну</div>', unsafe_allow_html=True)
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        cr  = st.number_input("🍞 CR — г вуглеводів на 1 ОД", 5, 30, int(ip["cr"]), step=1, key="ins_cr")
    with ic2:
        isf = st.number_input("📉 ISF — ммоль/л на 1 ОД", 0.5, 8.0, float(ip["isf"]), step=0.1, key="ins_isf")
    with ic3:
        iob_dur = st.number_input("⏳ Тривалість дії (год)", 2.0, 8.0, float(ip["iob_duration"]), step=0.5, key="ins_dur")

    if cr != ip["cr"] or isf != ip["isf"] or iob_dur != ip["iob_duration"]:
        ip.update({"cr": cr, "isf": isf, "iob_duration": iob_dur})
        save_data()

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Вхідні дані</div>', unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)
    with ic1:
        g_cur = st.number_input("🩸 Поточна глюкоза (ммоль/л)", 0.5, 35.0, 5.5, step=0.1, key="ins_gcur")
    with ic2:
        g_tgt = st.number_input("🎯 Цільова глюкоза (ммоль/л)", 3.0, 12.0,
                                  st.session_state.user_profile["target_min"], step=0.1, key="ins_gtgt")

    label, key_z, color, msg = get_zone(g_cur)
    trend = glucose_trend_arrow(st.session_state.glucose_logs)
    st.markdown(
        f'<span class="badge badge-{key_z}">{label}</span> '
        f'{trend_arrow_html(trend)} '
        f'<span style="font-size:12px;color:#6b7280;margin-left:8px">{msg}</span>',
        unsafe_allow_html=True)

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⏳ Активний інсулін (IOB)</div>', unsafe_allow_html=True)

    active_doses   = ip.get("active_doses", [])
    cumulative_iob = total_iob(active_doses, iob_dur)

    if active_doses:
        st.markdown(f"""
        <div class="iob-bar" style="margin-bottom:12px">
          <div style="display:flex;justify-content:space-between;margin-bottom:8px">
            <span style="font-size:12px;color:#9ca3af">📊 Накопичений IOB ({len(active_doses)} доз)</span>
            <span style="font-family:'DM Mono',monospace;color:#60a5fa;font-size:15px;font-weight:600">{cumulative_iob} ОД</span>
          </div>
          <div style="background:rgba(255,255,255,.05);border-radius:6px;height:5px;overflow:hidden">
            <div style="background:linear-gradient(90deg,#60a5fa,#e8506a);
                        width:{min(100, cumulative_iob/max(sum(d['units'] for d in active_doses),0.1)*100):.0f}%;
                        height:100%;border-radius:6px;transition:width 0.5s"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    ibc1, ibc2, ibc3 = st.columns(3)
    with ibc1:
        iob_units = st.number_input("Доза (ОД)", 0.0, 30.0, 0.0, step=0.5, key="iob_u")
    with ibc2:
        iob_min = st.number_input("Хвилин тому", 0, 480, 0, step=5, key="iob_m")
    with ibc3:
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
                        height:100%;border-radius:6px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    auto_tod = auto_tod_key()
    st.markdown(f'<div class="section-title">🕐 Час доби (авто: {auto_tod})</div>', unsafe_allow_html=True)
    tod_key = st.radio("Час доби", list(TOD_ISF_FACTORS.keys()),
                        index=list(TOD_ISF_FACTORS.keys()).index(auto_tod),
                        horizontal=True, key="ins_tod", label_visibility="collapsed")
    tod_factor = TOD_ISF_FACTORS[tod_key]
    adj_isf    = isf * tod_factor
    st.caption(f"ISF скоригований: {isf:.1f} × {tod_factor:.2f} = {adj_isf:.2f} ммоль/л на ОД")

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    with st.expander("🔬 Розширений болюс (білки + жири — Варшавський метод)"):
        st.markdown('<div style="font-size:12px;color:#6b7280;margin-bottom:12px">Враховує вплив білків та жирів на глюкозу через 2–5 год після їжі</div>', unsafe_allow_html=True)
        totals_now = get_totals()
        eb1, eb2, eb3 = st.columns(3)
        with eb1:
            eb_carbs   = st.number_input("Вуглеводи (г)", 0.0, 500.0, float(totals_now["carbs"]), step=1.0, key="eb_c")
        with eb2:
            eb_protein = st.number_input("Білки (г)", 0.0, 200.0, float(totals_now["protein"]), step=1.0, key="eb_p")
        with eb3:
            eb_fat     = st.number_input("Жири (г)", 0.0, 200.0, float(totals_now["fat"]), step=1.0, key="eb_f")

        if st.button("🔢 Розрахувати розширений болюс", key="btn_eb", use_container_width=True):
            eb = extended_bolus(eb_carbs, eb_protein, eb_fat, cr)
            st.markdown(f"""
            <div class="card card-accent" style="padding:20px">
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;text-align:center">
                <div class="kpi">
                  <span class="kpi-l">🍽️ На вуглеводи</span>
                  <span class="kpi-v" style="color:#e8506a">{eb["carb_dose"]} ОД</span>
                </div>
                <div class="kpi">
                  <span class="kpi-l">🥩 На БЖ (відстрочений)</span>
                  <span class="kpi-v" style="color:#a78bfa">{eb["ext_dose"]} ОД</span>
                </div>
                <div class="kpi">
                  <span class="kpi-l">💉 Всього</span>
                  <span class="kpi-v" style="color:#fbbf24">{eb["rounded"]} ОД</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    totals = get_totals()
    carbs  = totals["carbs"]

    if st.button("⚡ Розрахувати дозу болюсу", use_container_width=True, key="ins_calc"):
        meal_dose = carbs / cr if cr > 0 else 0
        corr_dose = (g_cur - g_tgt) / adj_isf if adj_isf > 0 else 0
        effective_iob = max(cumulative_iob, computed_iob)
        total     = max(0.0, meal_dose + corr_dose - effective_iob)
        rounded   = round(total * 2) / 2

        st.markdown(dose_card(total, rounded, meal_dose, corr_dose, effective_iob, carbs, tod_factor),
                    unsafe_allow_html=True)

        dose_col1, dose_col2 = st.columns(2)
        with dose_col1:
            if st.button(f"💉 Записати {rounded:.1f} ОД як введену дозу", key="log_dose_btn"):
                new_dose = {
                    "units": rounded, "timestamp": datetime.now().isoformat(),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "carbs": carbs, "glucose_before": g_cur,
                    "note": f"{get_meal_type(datetime.now().hour)}",
                }
                ip.setdefault("active_doses", []).append(new_dose)
                st.session_state.dose_log.append({
                    **new_dose, "calculated": total, "rounded": rounded,
                    "meal_dose": meal_dose, "corr_dose": corr_dose,
                    "iob": effective_iob, "glucose_after": None,
                })
                cutoff = datetime.now() - timedelta(hours=iob_dur + 1)
                ip["active_doses"] = [d for d in ip["active_doses"]
                    if datetime.fromisoformat(d["timestamp"]) > cutoff]
                save_data()
                st.success(f"✅ Доза {rounded:.1f} ОД записана!")
                st.rerun()

        if g_cur < 4.0:
            st.error("🚨 ГІПОГЛІКЕМІЯ! НЕ вводьте інсулін! З'їжте 15 г швидких вуглеводів.")
        elif g_cur >= 14.0:
            st.warning("⚠️ Дуже високий рівень! Перевірте кетони.")

        # Prediction chart
        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Прогноз глюкози після болюсу (4 год)</div>', unsafe_allow_html=True)

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
            marker=dict(size=12, color="#fbbf24", line=dict(color="#080a0e", width=2)),
            name="Зараз", hovertemplate="Зараз: %{y:.1f}<extra></extra>"
        ))
        fig.update_layout(height=280, showlegend=False,
                          xaxis_title="Хвилини після болюсу", yaxis_title="ммоль/л",
                          **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚠️ Прогноз — спрощена модель. Реальний результат залежить від багатьох факторів.")
    else:
        st.info(f"📋 Вуглеводів у поточному прийомі: **{carbs} г ({totals['bu']} ХО)**. Натисніть «Розрахувати».")


def tab_analytics():
    PLOTLY_THEME = st.session_state.get("_plotly_theme", {})
    glogs = st.session_state.glucose_logs
    dtots = st.session_state.daily_totals
    C = _theme_colors()

    if not glogs and not dtots:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:56px;color:{C['text_faint']}">
          <div style="font-size:2rem;margin-bottom:12px">📊</div>
          <div>Почніть записувати глюкозу та зберігайте прийоми їжі,<br>щоб тут з'явилася аналітика.</div>
        </div>""", unsafe_allow_html=True)
        return

    a1, a2, a3 = st.tabs(["🩸 Глюкоза & TIR", "🌍 AGP 24-год", "🍽️ Прийоми їжі"])

    with a1:
        _analytics_glucose(glogs, PLOTLY_THEME, C)
    with a2:
        _analytics_agp(glogs, PLOTLY_THEME, C)
    with a3:
        _analytics_meals(dtots, PLOTLY_THEME, C)


def _analytics_glucose(logs: list, PLOTLY_THEME: dict, C: dict):
    if not logs:
        st.info("Додайте вимірювання глюкози через бічну панель")
        return

    stats  = tir_stats(logs)
    tmin   = st.session_state.user_profile["target_min"]
    tmax   = st.session_state.user_profile["target_max"]

    st.markdown(hba1c_card(stats["hba1c"], stats["gmi"], stats["avg"],
                            stats["target"], stats["cv"]),
                unsafe_allow_html=True)

    now   = datetime.now()
    week1 = [e for e in logs if now - timedelta(days=7) <= datetime.fromisoformat(e["timestamp"]) <= now]
    week2 = [e for e in logs if now - timedelta(days=14) <= datetime.fromisoformat(e["timestamp"]) <= now - timedelta(days=7)]

    if week1 and week2:
        w1s = tir_stats(week1)
        w2s = tir_stats(week2)
        delta_tir = w1s["target"] - w2s["target"]
        delta_avg = round(w1s["avg"] - w2s["avg"], 1)
        d_color_tir = "#34d399" if delta_tir >= 0 else "#f87171"
        d_color_avg = "#34d399" if delta_avg <= 0 else "#f87171"
        st.markdown(f"""
        <div class="card card-sm" style="display:flex;gap:24px;align-items:center;padding:14px 20px">
          <span style="font-size:11px;color:{C['text_faint']};text-transform:uppercase;letter-spacing:.08em">📅 Цей тиждень vs минулий</span>
          <span style="font-size:13px">TIR: <b style="color:{d_color_tir}">{w1s["target"]}%</b> <span style="color:{C['text_faint']};font-size:11px">({'+' if delta_tir>=0 else ''}{delta_tir}%)</span></span>
          <span style="font-size:13px">Середня: <b style="color:{d_color_avg}">{w1s["avg"]} мм</b></span>
          <span style="font-size:13px">CV%: <b style="color:#a78bfa">{w1s["cv"]}%</b></span>
        </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1.6, 1])

    with c1:
        times  = [e["timestamp"][:16].replace("T", " ") for e in logs[-50:]]
        levels = [e["level"] for e in logs[-50:]]
        colors = [get_zone(lv)[2] for lv in levels]

        fig = go.Figure()
        fig.add_hrect(y0=tmin, y1=tmax, fillcolor="rgba(52,211,153,0.05)", line_width=0,
                      annotation_text=f"Ціль {tmin}–{tmax}", annotation_font_size=10,
                      annotation_font_color="#34d399")
        fig.add_hrect(y0=0, y1=4.0, fillcolor="rgba(248,113,113,0.03)", line_width=0)
        fig.add_trace(go.Scatter(
            x=times, y=levels, mode="lines+markers",
            line=dict(color="#60a5fa", width=2, shape="spline"),
            marker=dict(size=8, color=colors, line=dict(color="#080a0e", width=1.5)),
            hovertemplate="<b>%{x}</b><br>%{y:.1f} ммоль/л<extra></extra>"
        ))
        fig.update_layout(title="🩸 Глюкоза в часі (останні 50 вимірювань)", height=300, **PLOTLY_THEME)
        fig.update_xaxes(tickangle=-35, nticks=8)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tir_vals   = [stats["hypo"], stats["low"], stats["target"], stats["high"], stats["very_high"]]
        tir_labels = ["Гіпо <4.0", "Низький 4–4.5", "Норма 4.5–7.8", "Висок. 7.8–14", "Дуже висок. >14"]
        tir_colors = ["#f87171","#fb923c","#34d399","#fbbf24","#ef4444"]

        fig2 = go.Figure(go.Pie(
            labels=tir_labels, values=tir_vals, hole=0.60,
            marker=dict(colors=tir_colors, line=dict(color="#080a0e", width=2)),
            textinfo="percent", textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>%{value}%<extra></extra>"
        ))
        fig2.add_annotation(
            text=f"<b>{stats['target']}%</b><br>в нормі",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#34d399", family="DM Mono")
        )
        fig2.update_layout(
            title="🎯 Time-in-Range", height=300, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C["text_muted"]), margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

        legend_html = '<div class="tir-legend">'
        for label, color, val in zip(tir_labels, tir_colors, tir_vals):
            legend_html += f'<div class="tir-item"><div class="tir-dot" style="background:{color}"></div>{label}: {val}%</div>'
        legend_html += '</div>'
        st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    levels_all = [e["level"] for e in logs]

    with c3:
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=levels_all, nbinsx=25, marker_color="#60a5fa",
            marker_line=dict(color="#080a0e", width=1),
            hovertemplate="<b>%{x:.1f} мм</b><br>%{y} вимірювань<extra></extra>"
        ))
        fig3.add_vrect(x0=tmin, x1=tmax, fillcolor="rgba(52,211,153,0.07)", line_width=0,
                       annotation_text="Норма", annotation_font_color="#34d399", annotation_font_size=10)
        fig3.update_layout(title="📊 Розподіл глюкози", height=260,
                           xaxis_title="ммоль/л", yaxis_title="Вимірювань", **PLOTLY_THEME)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.metric("Всього вимірювань", stats["n"])
        st.metric("Середня глюкоза", f"{stats['avg']} ммоль/л")
        st.metric("CV% (варіабельність)", f"{stats['cv']}%",
                  delta="стабільно ✓" if stats["cv"] < 36 else "варіабельно ⚠",
                  delta_color="normal" if stats["cv"] < 36 else "inverse")
        st.metric("GMI (розрахунковий)", f"{stats['gmi']}%")

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 Аналіз патернів</div>', unsafe_allow_html=True)
    _pattern_insights(logs, tmin, tmax)


def _pattern_insights(logs: list, tmin: float, tmax: float):
    from collections import defaultdict
    by_hour = defaultdict(list)
    for e in logs:
        try:
            h = datetime.fromisoformat(e["timestamp"]).hour
            by_hour[h].append(e["level"])
        except Exception:
            pass

    periods = {
        "🌅 Ранок (6–10)":   list(range(6, 10)),
        "☀️ День (10–17)":   list(range(10, 17)),
        "🌆 Вечір (17–21)":  list(range(17, 21)),
        "🌙 Ніч (21–6)":     list(range(21, 24)) + list(range(0, 6)),
    }

    insights = []
    for period, hours in periods.items():
        vals = [v for h in hours for v in by_hour.get(h, [])]
        if len(vals) < 3:
            continue
        avg = np.mean(vals)
        tir = sum(1 for v in vals if tmin <= v <= tmax) / len(vals) * 100

        if avg > tmax + 1.5:
            insights.append(f"<div class='insight-card'><b>{period}</b>: Середня {avg:.1f} мм — систематично висока. Можливо, потрібна корекція CR.</div>")
        elif avg < tmin - 0.5:
            insights.append(f"<div class='insight-card' style='background:rgba(248,113,113,.05);border-color:rgba(248,113,113,.2)'><b>{period}</b>: Середня {avg:.1f} мм — ризик гіпоглікемії.</div>")
        elif tir >= 80:
            insights.append(f"<div class='insight-card' style='background:rgba(52,211,153,.05);border-color:rgba(52,211,153,.2)'><b>{period}</b>: TIR {tir:.0f}% — чудовий контроль! 🎉</div>")

    if insights:
        for ins in insights:
            st.markdown(ins, unsafe_allow_html=True)
    else:
        st.markdown('<div class="insight-card">Поки недостатньо даних для патернів. Додавайте більше вимірювань!</div>', unsafe_allow_html=True)


def _analytics_agp(logs: list, PLOTLY_THEME: dict, C: dict):
    if not logs or len(logs) < 5:
        st.info("Потрібно більше вимірювань для AGP (мін. 5)")
        return

    st.markdown('<div class="section-title">🌍 AGP — Ambulatory Glucose Profile</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;color:{C["text_muted"]};margin-bottom:16px">Накладання всіх вимірювань по годинах доби</div>', unsafe_allow_html=True)

    from collections import defaultdict
    by_hour = defaultdict(list)
    for e in logs:
        try:
            h = datetime.fromisoformat(e["timestamp"]).hour
            by_hour[h].append(e["level"])
        except Exception:
            pass

    hours_sorted = sorted(by_hour.keys())
    if len(hours_sorted) < 3:
        st.info("Недостатньо покриття годин доби для AGP")
        return

    h_list = hours_sorted
    p10 = [np.percentile(by_hour[h], 10) for h in h_list]
    p25 = [np.percentile(by_hour[h], 25) for h in h_list]
    p50 = [np.percentile(by_hour[h], 50) for h in h_list]
    p75 = [np.percentile(by_hour[h], 75) for h in h_list]
    p90 = [np.percentile(by_hour[h], 90) for h in h_list]
    tmin = st.session_state.user_profile["target_min"]
    tmax = st.session_state.user_profile["target_max"]

    fig = go.Figure()
    fig.add_hrect(y0=tmin, y1=tmax, fillcolor="rgba(52,211,153,0.06)", line_width=0)
    fig.add_hrect(y0=0, y1=4.0, fillcolor="rgba(248,113,113,0.04)", line_width=0)
    fig.add_hrect(y0=13.9, y1=30, fillcolor="rgba(248,113,113,0.04)", line_width=0)

    fig.add_trace(go.Scatter(x=h_list + h_list[::-1], y=p90 + p10[::-1],
        fill='toself', fillcolor='rgba(96,165,250,0.07)',
        line=dict(color='rgba(96,165,250,0)', width=0), name='10–90 перцентиль', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=h_list + h_list[::-1], y=p75 + p25[::-1],
        fill='toself', fillcolor='rgba(96,165,250,0.18)',
        line=dict(color='rgba(96,165,250,0)', width=0), name='25–75 перцентиль', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=h_list, y=p50, mode='lines',
        line=dict(color='#60a5fa', width=2.5), name='Медіана',
        hovertemplate='<b>%{x}:00</b><br>Медіана: %{y:.1f} мм<extra></extra>'))
    fig.add_hline(y=tmin, line=dict(color='#34d399', width=1, dash='dot'))
    fig.add_hline(y=tmax, line=dict(color='#34d399', width=1, dash='dot'))
    fig.add_hline(y=4.0,  line=dict(color='#f87171', width=1, dash='dot'))

    hour_labels = [f"{h}:00" for h in h_list]
    fig.update_layout(height=360,
        xaxis=dict(tickmode='array', tickvals=h_list, ticktext=hour_labels, title='Година доби'),
        yaxis=dict(title='ммоль/л', range=[2, 20]),
        legend=dict(orientation='h', y=-0.15, font=dict(size=10, color=C["text_muted"])),
        **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)


def _analytics_meals(dtots: dict, PLOTLY_THEME: dict, C: dict):
    if not dtots:
        st.info("Збережіть кілька прийомів їжі, щоб побачити аналітику")
        return

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
        fig.add_trace(go.Scatter(x=dates, y=bus, mode="lines+markers", name="ХО/день",
            line=dict(color="#e8506a", width=2, shape="spline"),
            marker=dict(size=6, color="#e8506a"),
            fill="tozeroy", fillcolor="rgba(232,80,106,0.06)",
            hovertemplate="<b>%{x}</b><br>%{y:.1f} ХО<extra></extra>"))
        fig.update_layout(title="📈 ХО по днях", height=260, **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        freq = st.session_state.product_freq
        if freq:
            top10 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8]
            fig4  = go.Figure(go.Bar(
                x=[v for _, v in top10], y=[n for n, _ in top10],
                orientation="h", marker_color="#8b5cf6", marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>%{x}x<extra></extra>"
            ))
            fig4.update_layout(title="🏆 Топ продуктів", height=260, **PLOTLY_THEME, xaxis_title="Разів")
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📅 Зведення за останній тиждень</div>', unsafe_allow_html=True)
    end   = datetime.now()
    start = end - timedelta(days=7)
    week  = {d: ms for d, ms in dtots.items() if start <= datetime.fromisoformat(d) <= end}
    if week:
        total_m = sum(len(ms) for ms in week.values())
        all_bu  = [sum(m["totals"]["bu"] for m in ms) for ms in week.values()]
        avg_bu  = round(sum(all_bu) / len(all_bu), 1)
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("Прийомів", total_m)
        wc2.metric("Середньо ХО/день", avg_bu)
        wc3.metric("Макс ХО/день", round(max(all_bu), 1))
        wc4.metric("Мін ХО/день", round(min(all_bu), 1))
    else:
        st.info("Немає даних за останні 7 днів")


def tab_history():
    C = _theme_colors()
    dtots = st.session_state.daily_totals

    if not dtots:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:56px;color:{C['text_faint']}">
          <div style="font-size:2rem;margin-bottom:12px">📋</div>
          Збережіть перший прийом їжі, щоб він з'явився тут.
        </div>""", unsafe_allow_html=True)
        return

    all_dates = sorted(dtots.keys(), reverse=True)
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        show_days = st.slider("Показати днів", 3, min(60, len(all_dates)), 14, key="hist_days")
    with c2:
        if st.button("📥 Експорт JSON", key="exp_json"):
            j = json.dumps({
                "profile":      st.session_state.user_profile,
                "daily_totals": st.session_state.daily_totals,
                "glucose_logs": st.session_state.glucose_logs,
                "dose_log":     st.session_state.dose_log,
                "ketone_logs":  st.session_state.ketone_logs,
                "exported":     datetime.now().isoformat(),
            }, ensure_ascii=False, indent=2)
            st.download_button("⬇️ Завантажити JSON", data=j,
                file_name=f"glypro_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json", key="dl_json")
    with c3:
        if st.session_state.glucose_logs:
            gl_df  = pd.DataFrame(st.session_state.glucose_logs)
            gl_csv = gl_df.to_csv(index=False)
            st.download_button("📊 Глюкоза CSV", data=gl_csv,
                file_name=f"glucose_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", key="dl_gcsv")

    for date in all_dates[:show_days]:
        meals     = dtots[date]
        day_bu    = round(sum(m["totals"]["bu"]    for m in meals), 1)
        day_carbs = round(sum(m["totals"]["carbs"] for m in meals), 1)
        day_cal   = round(sum(m["totals"]["cal"]   for m in meals), 0)
        day_glogs = [e for e in st.session_state.glucose_logs if e.get("date") == date]
        g_summary = ""
        if day_glogs:
            avg_g = round(np.mean([e["level"] for e in day_glogs]), 1)
            g_summary = f"  ·  🩸 {avg_g} мм ср."

        with st.expander(f"📅  {date}  ·  {len(meals)} прийомів  ·  {day_bu} ХО  ·  {day_carbs} г{g_summary}"):
            for meal in meals:
                ts  = meal["timestamp"][11:16]
                mt  = meal.get("meal_type", "")
                t   = meal["totals"]
                n_p = len(meal.get("data", []))
                st.markdown(f"""
                <div class="card card-sm" style="margin-bottom:8px">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                      <span style="font-size:11px;color:{C['text_faint']}">🕐 {ts} &nbsp; {mt}</span>
                      <span style="font-size:11px;color:{C['text_muted']};margin-left:12px">{n_p} продуктів</span>
                    </div>
                    <div style="text-align:right">
                      <span style="font-family:'DM Mono',monospace;color:#e8506a;font-size:1rem;font-weight:500">{t["bu"]} ХО</span>
                      <span style="font-size:11px;color:{C['text_muted']};margin-left:8px">{t["carbs"]} г · {int(t["cal"])} ккал</span>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

                if meal.get("data"):
                    df   = pd.DataFrame(meal["data"])
                    cols = [c for c in ["Продукт","Вага","Вугл.","ХО","Ккал","Білки","Жири"] if c in df.columns]
                    st.dataframe(df[cols], use_container_width=True,
                                 hide_index=True, height=150,
                                 key=f"hist_{date}_{meal['timestamp']}")

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    if st.checkbox("🗑️ Показати кнопку очищення журналу"):
        if st.button("❌ Очистити весь журнал", key="clr_hist_btn"):
            st.session_state.daily_totals = {}
            save_data(); st.rerun()


def tab_doses():
    C = _theme_colors()
    st.markdown("""
    <div class="warn-bar">
      ⚠️ Журнал доз — тільки для ознайомлення та самоконтролю.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:16px">💉 Журнал введених доз</div>', unsafe_allow_html=True)

    with st.expander("➕ Вручну додати дозу"):
        dc1, dc2, dc3, dc4 = st.columns([1, 1, 2, 1])
        with dc1:
            d_units = st.number_input("Доза (ОД)", 0.5, 100.0, 1.0, step=0.5, key="d_units")
        with dc2:
            d_glucose = st.number_input("Глюкоза до (мм)", 0.5, 35.0, 5.5, step=0.1, key="d_glucose")
        with dc3:
            d_note = st.text_input("Примітка", placeholder="Сніданок, корекція...", key="d_note")
        with dc4:
            st.markdown("<br/>", unsafe_allow_html=True)
            if st.button("💉 Записати", key="btn_d_add", use_container_width=True):
                ip = st.session_state.insulin_profile
                st.session_state.dose_log.append({
                    "units": d_units, "glucose_before": d_glucose,
                    "timestamp": datetime.now().isoformat(),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "note": d_note, "manual": True,
                    "glucose_after": None, "carbs": 0,
                })
                ip.setdefault("active_doses", []).append({
                    "units": d_units, "timestamp": datetime.now().isoformat(),
                })
                save_data()
                st.success(f"✅ {d_units} ОД записано!")
                st.rerun()

    dose_log = st.session_state.dose_log
    if not dose_log:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:48px;color:{C['text_faint']};font-style:italic">
          💉 Дози ще не записувалися.
        </div>""", unsafe_allow_html=True)
        return

    # Summary metrics
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_doses = [d for d in dose_log if d.get("date") == today_str]
    total_today = round(sum(d["units"] for d in today_doses), 1)
    dm1, dm2, dm3 = st.columns(3)
    dm1.metric("Доз сьогодні", len(today_doses))
    dm2.metric("ОД сьогодні", total_today)
    dm3.metric("Всього у журналі", len(dose_log))

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    recent_doses = sorted(dose_log, key=lambda x: x.get("timestamp",""), reverse=True)[:30]
    for dose in recent_doses:
        ts   = dose.get("timestamp","")
        date = ts[:10] if ts else ""
        time = ts[11:16] if len(ts) > 15 else ""
        note = dose.get("note","")
        g_bef = dose.get("glucose_before")
        carbs = dose.get("carbs")
        manual_tag = '<span class="badge badge-info" style="font-size:10px">вручну</span>' if dose.get("manual") else ""
        st.markdown(f"""
        <div class="dose-item">
          <div>
            <span style="font-size:12px;color:{C['text_faint']}">{date} {time}</span>
            <span style="font-size:12px;color:{C['text_muted']};margin-left:10px">{note}</span>
            {manual_tag}
          </div>
          <div style="display:flex;gap:16px;align-items:center">
            {f'<span style="font-size:11px;color:{C["text_muted"]}">🩸 {g_bef} мм</span>' if g_bef else ''}
            {f'<span style="font-size:11px;color:{C["text_muted"]}">🍽️ {carbs} г</span>' if carbs else ''}
            <span style="font-family:\'DM Mono\',monospace;color:#e8506a;font-size:1rem;font-weight:600">{dose['units']:.1f} ОД</span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    if st.checkbox("🗑️ Показати очищення доз"):
        if st.button("❌ Очистити журнал доз", key="clr_dose"):
            st.session_state.dose_log = []
            st.session_state.insulin_profile["active_doses"] = []
            save_data(); st.rerun()


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
            activity     = st.selectbox("Активність", ["low","medium","high"],
                                         index=["low","medium","high"].index(p.get("activity","medium")))
            insulin_type = st.selectbox("Тип інсуліну", ["rapid","short","intermediate","long"],
                                         index=["rapid","short","intermediate","long"].index(p.get("insulin_type","rapid")))

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
            cr_auto  = round(500 / tdd, 1)
            isf_auto = round(1700 / tdd / 18, 2)
            st.session_state.insulin_profile.update({"cr": cr_auto, "isf": isf_auto})
            save_data()
            st.success(f"✅ Збережено! CR = {cr_auto} г/ОД, ISF = {isf_auto} ммоль/ОД")

    tdd = st.session_state.user_profile.get("tdd", 35.0)
    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧮 Автоматично розраховані параметри (правило 500/1700)</div>', unsafe_allow_html=True)
    ac1, ac2, ac3, ac4 = st.columns(4)
    ac1.metric("CR (500 / TDD)", f"{round(500/tdd,1)} г/ОД")
    ac2.metric("ISF (1700 / TDD / 18)", f"{round(1700/tdd/18,2)} мм/ОД")
    ac3.metric("Базальна доза", f"{round(tdd*0.5,1)} ОД/добу")
    ac4.metric("Болюсна доза", f"{round(tdd*0.5,1)} ОД/добу")

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    with st.expander("⚠️ Небезпечна зона — видалення даних"):
        st.warning("Ці дії незворотні!")
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            if st.button("🗑️ Очистити прийом"):
                st.session_state.meal_data = []; save_data(); st.success("✅ Прийом очищено")
        with dc2:
            if st.button("🗑️ Очистити глюкозу"):
                st.session_state.glucose_logs = []; save_data(); st.success("✅ Глюкозу очищено")
        with dc3:
            if st.button("💥 Скинути ВСЕ"):
                for key in ["meal_data","daily_totals","glucose_logs","meal_patterns",
                            "meal_templates","product_freq","product_history","ketone_logs","dose_log"]:
                    st.session_state[key] = [] if isinstance(st.session_state[key], list) else {}
                st.session_state.insulin_profile["active_doses"] = []
                save_data(); st.success("✅ Усі дані скинуто"); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="GlyPro v7",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session()
    inject_css()
    render_sidebar()

    mode = st.session_state.app_mode

    if mode == "lite":
        render_lite()
    else:
        # Pro mode header
        C = _theme_colors()
        hc1, hc2 = st.columns([3, 2])
        with hc1:
            name_greeting = f", {st.session_state.user_profile['name']}" \
                            if st.session_state.user_profile.get("name") else ""
            st.markdown(
                '<div class="glyco-title" style="font-size:2.6rem">GlyPro <span style="font-size:1rem;opacity:.5">Pro</span></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:13px;color:{C["text_faint"]};margin-top:2px">'
                f'Щоденник діабетика 1 типу{name_greeting} · v{APP_VERSION}</div>',
                unsafe_allow_html=True)
        with hc2:
            t    = get_totals()
            glogs = st.session_state.glucose_logs
            last_g = glogs[-1]["level"] if glogs else None
            trend  = glucose_trend_arrow(glogs)
            trend_cls_map = {"↑↑": "trend-up2", "↑": "trend-up", "→": "trend-flat",
                             "↓": "trend-down", "↓↓": "trend-down2"}
            trend_cls = trend_cls_map.get(trend, "trend-flat")
            klogs = st.session_state.ketone_logs
            last_k = klogs[-1] if klogs else None
            st.markdown(f"""
            <div style="display:flex;gap:10px;justify-content:flex-end;align-items:center;margin-top:8px;flex-wrap:wrap">
              {f'<span class="badge badge-{get_zone(last_g)[1]}">{last_g} мм <span class="{trend_cls}">{trend}</span></span>' if last_g else ''}
              {f'<span class="badge badge-info">{t["bu"]} ХО</span>' if t["bu"] > 0 else ''}
              {f'<span class="badge badge-info">{len(st.session_state.meal_data)} продуктів</span>' if st.session_state.meal_data else ''}
              {f'<span class="badge badge-purple">🧪 {last_k["value"]} кет.</span>' if last_k else ''}
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="divider" style="margin:12px 0 20px"/>', unsafe_allow_html=True)

        # ── AI BAR ──
        render_ai_bar()

        t1, t2, t3, t4, t5, t6 = st.tabs([
            "🍽️ Прийом їжі",
            "💉 Інсулін",
            "📊 Аналітика",
            "📋 Журнал",
            "💊 Дози",
            "⚙️ Налаштування",
        ])

        with t1: tab_meal()
        with t2: tab_insulin()
        with t3: tab_analytics()
        with t4: tab_history()
        with t5: tab_doses()
        with t6: tab_settings()

        # Footer
        st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
        total_days = len(st.session_state.daily_totals)
        total_g    = len(st.session_state.glucose_logs)
        total_d    = len(st.session_state.dose_log)
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    font-size:11px;color:{C['text_faint']};padding-bottom:8px;flex-wrap:wrap;gap:8px">
          <div>GlyPro v{APP_VERSION} · <span style="color:#e8506a">♥</span> Зроблено з турботою · <span>Тільки для ознайомлення</span></div>
          <div style="display:flex;gap:12px">
            <span>📅 {total_days} днів</span>
            <span>🩸 {total_g} вимірювань</span>
            <span>💉 {total_d} доз</span>
          </div>
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
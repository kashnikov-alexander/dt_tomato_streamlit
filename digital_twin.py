import streamlit as st
import pandas as pd
import numpy as np
import joblib
import optuna
import json

# Настройка страницы (широкий формат)
st.set_page_config(page_title="Цифровой двойник теплицы", page_icon="🌱", layout="wide")

# Отключаем лишние логи Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Кэширование загрузки моделей
@st.cache_resource
def load_models():
    try:
        gp = joblib.load("gp_model.pkl")
        gb = joblib.load("gb_model.pkl")
        scaler = joblib.load("scaler.pkl")
        imputer = joblib.load("imputer.pkl")
        features = scaler.feature_names_in_.tolist()
        return gp, gb, scaler, imputer, features
    except FileNotFoundError as e:
        st.error(f"Ошибка загрузки моделей: Не найден файл - {e.filename}.")
        return None, None, None, None, []

# 2. Конфигурация 19 управляемых признаков
SLIDER_CONFIG = {
    "lamp_hours": {"label": "Работа ламп (часы)", "step": 1},
    "total_illuminance": {"label": "Общая освещенность (Дж/см²)", "step": 50},
    "irrigation_illuminance": {"label": "Освещенность при поливе (Дж/см²)", "step": 50},
    "avg_outdoor_temp": {"label": "Ср. уличная темп. (°C)", "step": 0.5},
    "avg_greenhouse_temp": {"label": "Ср. темп. в теплице (°C)", "step": 0.5},
    "avg_day_temp": {"label": "Ср. дневная темп. (°C)", "step": 0.5},
    "avg_night_temp": {"label": "Ср. ночная темп. (°C)", "step": 0.5},
    "rail_pipes_avg_day": {"label": "Рельсовые трубы день (°C)", "step": 0.5},
    "rail_pipes_avg_night": {"label": "Рельсовые трубы ночь (°C)", "step": 0.5},
    "humidity_day": {"label": "Влажность днем (%)", "step": 1},
    "humidity_night": {"label": "Влажность ночью (%)", "step": 1},
    "avg_co2": {"label": "Ср. CO2 (ppm)", "step": 1},
    "drainage_percent": {"label": "Дренаж (%)", "step": 1},
    "consumed_volume": {"label": "Потребление воды (мл)", "step": 10},
    "drip_EC": {"label": "Капельница EC (мСм/см)", "step": 0.1},
    "drip_pH": {"label": "Капельница pH", "step": 0.1},
    "substrate_EC": {"label": "Электропроводность (ЕС)", "step": 0.1},
    "substrate_moisture": {"label": "Влажность субстрата (%)", "step": 1},
    "substrate_pH": {"label": "Субстрат pH", "step": 0.1}
}

# Загружаем модели
gp_model, gb_model, scaler_p, imp_p, feature_names = load_models()
if not feature_names:
    st.stop()

# Загружаем границы из JSON
try:
    with open("feature_bounds.json", "r", encoding="utf-8") as f:
        DATA_BOUNDS = json.load(f)
except FileNotFoundError:
    st.error("Файл feature_bounds.json не найден. Запустите prepare_data.py")
    st.stop()

# ==============================================================================
# БЛОК ПРИМЕНЕНИЯ РЕЗУЛЬТАТОВ ОПТУНЫ (Срабатывает при перезапуске страницы)
# ==============================================================================
if st.session_state.get("apply_optuna_updates", False):
    saved_params = st.session_state.get("optimized_slider_values", {})
    for param_name, param_value in saved_params.items():
        if param_name in st.session_state:
            st.session_state[param_name] = float(param_value)
    st.session_state["apply_optuna_updates"] = False
# ==============================================================================

st.title("🌱 Цифровой двойник теплицы")
st.markdown("Настройте параметры среды или запустите консенсус-оптимизацию.")

# 3. Отрисовка интерфейса (2 колонки, строгий порядок)
st.subheader("Параметры среды")

# Задаем точный порядок вывода слайдеров
UI_ORDER = [
    "lamp_hours", "total_illuminance", "avg_outdoor_temp", "avg_greenhouse_temp",
    "avg_day_temp", "avg_night_temp", "rail_pipes_avg_day", "rail_pipes_avg_night",
    "humidity_day", "humidity_night",
    "avg_co2", "drainage_percent", "consumed_volume", "irrigation_illuminance",
    "drip_EC", "drip_pH", "substrate_EC", "substrate_moisture", "substrate_pH"
]

# ВАЖНО: Инициализируем значения в памяти для ВСЕХ признаков из модели
# Это защитит от ошибки, если в модели есть параметр, которого нет в списке UI_ORDER
for _fname in feature_names:
    if _fname not in st.session_state:
        _bounds = DATA_BOUNDS.get(_fname, {"min": 0, "max": 100})
        st.session_state[_fname] = round((_bounds["min"] + _bounds["max"]) / 2, 2)

col1, col2 = st.columns(2)

# Отрисовываем ползунки в нужном порядке
for i, fname in enumerate(UI_ORDER):
    if fname not in feature_names:
        continue

    config = SLIDER_CONFIG.get(fname, {"label": fname, "step": 1})
    bounds = DATA_BOUNDS.get(fname, {"min": 0, "max": 100})

    # Первые 10 слайдеров в левую колонку, остальные в правую
    target_col = col1 if i < 10 else col2

    with target_col:
        st.slider(
            label=config["label"],
            min_value=float(bounds["min"]),
            max_value=float(bounds["max"]),
            step=float(config["step"]),
            key=fname
        )

# Теперь сбор данных безопасен, так как в st.session_state точно есть все ключи
inputs = {fname: st.session_state[fname] for fname in feature_names}

## 4. Мгновенный прогноз
df_input = pd.DataFrame([inputs]).reindex(columns=feature_names, fill_value=np.nan)
X_arr = imp_p.transform(df_input)
X_df = pd.DataFrame(X_arr, columns=feature_names)
X_scaled = scaler_p.transform(X_df)

pred_gb = float(gb_model.predict(X_df)[0])
pred_gp_arr, std_gp_arr = gp_model.predict(X_scaled, return_std=True)
pred_gp = float(pred_gp_arr[0])
std_val = float(std_gp_arr[0])

# Красивый вывод результатов через карточки
st.markdown(f"""
<div style="display: flex; gap: 30px; margin-top: 20px; margin-bottom: 30px;">
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 20px 30px; width: 250px; background-color: #fcfcfc;">
        <div style="font-size: 16px; color: #666; margin-bottom: 10px;">Gradient Boosting</div>
        <div style="font-size: 42px; font-weight: bold; color: #2ca02c; line-height: 1.1;">
            {pred_gb:.2f} <span style="font-size: 24px;">кг/м²</span>
        </div>
        <div style="font-size: 18px; color: gray; margin-top: 15px;">Точный прогноз</div>
    </div>
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 20px 30px; width: 250px; background-color: #fcfcfc;">
        <div style="font-size: 16px; color: #666; margin-bottom: 10px;">Gaussian Process</div>
        <div style="font-size: 42px; font-weight: bold; color: #1f77b4; line-height: 1.1;">
            {pred_gp:.2f} <span style="font-size: 24px;">кг/м²</span>
        </div>
        <div style="font-size: 16px; color: gray; margin-top: 15px;">
            ±1σ ({pred_gp - std_val:.2f} — {pred_gp + std_val:.2f})
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 5. Блок Консенсус-оптимизации
st.divider()
st.subheader("Консенсус-оптимизация (Optuna)")
st.markdown("Алгоритм ищет параметры, максимизирующие средний прогноз обеих моделей одновременно.")

if "consensus_result" not in st.session_state:
    st.session_state.consensus_result = None
if "is_optimizing" not in st.session_state:
    st.session_state.is_optimizing = False

if st.button("Найти оптимальные параметры"):
    st.session_state.consensus_result = None # <--- ОЧИЩАЕМ СТАРЫЙ РЕЗУЛЬТАТ
    st.session_state.is_optimizing = True
    st.rerun()

if st.session_state.get("is_optimizing", False):
    with st.spinner("Поиск лучших параметров..."):

        def prepare_input_data(param_values):
            input_data = {name: val for name, val in zip(feature_names, param_values)}
            df_in = pd.DataFrame([input_data])
            df_aligned = df_in.reindex(columns=feature_names, fill_value=np.nan)
            arr_transformed = imp_p.transform(df_aligned)
            df_transformed = pd.DataFrame(arr_transformed, columns=feature_names)
            scaled_data = scaler_p.transform(df_transformed)
            return df_transformed, scaled_data

        # ЕДИНАЯ ЦЕЛЕВАЯ ФУНКЦИЯ (Консенсус)
        def objective_consensus(trial):
            param_values = [
                trial.suggest_float(name, DATA_BOUNDS[name]["min"], DATA_BOUNDS[name]["max"])
                for name in feature_names
            ]
            df_transformed, scaled_data = prepare_input_data(param_values)

            pred_gb_val = float(gb_model.predict(df_transformed)[0])
            pred_gp_val = float(gp_model.predict(scaled_data, return_std=False)[0])

            # Ищем максимум среднего прогноза
            return (pred_gb_val + pred_gp_val) / 2.0

        # Запускаем ОДНО исследование
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42, n_startup_trials=50)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective_consensus, n_trials=300)

        # Извлекаем и округляем параметры
        best_params = {}
        for name in feature_names:
            val = study.best_params[name]
            step = SLIDER_CONFIG.get(name, {"step": 1})["step"]
            step_str = str(step)
            decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
            rounded_val = round((val - DATA_BOUNDS[name]["min"]) / step) * step + DATA_BOUNDS[name]["min"]
            best_params[name] = round(rounded_val, decimals)

        # Вычисляем прогнозы для найденных параметров
        df_final, scaled_final = prepare_input_data([best_params[n] for n in feature_names])
        final_gb = float(gb_model.predict(df_final)[0])
        final_gp_arr, final_std_arr = gp_model.predict(scaled_final, return_std=True)
        final_gp = float(final_gp_arr[0])
        final_std = float(final_std_arr[0])

        # Сохраняем результат
        st.session_state.consensus_result = {
            "params": best_params,
            "pred_gb": final_gb,
            "pred_gp": final_gp,
            "std_gp": final_std, # <--- ДОБАВИТЬ ЭТО
            "consensus": (final_gb + final_gp) / 2.0
        }

        # Обновляем ползунки
        st.session_state["optimized_slider_values"] = best_params
        st.session_state["apply_optuna_updates"] = True
        st.session_state.is_optimizing = False
        st.rerun()

# 6. Вывод результатов
if st.session_state.consensus_result:
    res = st.session_state.consensus_result

    # Безопасно достаем сигму. Если её нет (старый кэш), берем 0.0
    std_gp_val = res.get('std_gp', 0.0)
    lower_bound = res['pred_gp'] - std_gp_val
    upper_bound = res['pred_gp'] + std_gp_val

    st.success(f"Оптимизация успешно завершена! Ожидаемая урожайность: {res['consensus']:.2f} кг/м²")

    # Карточки для оптимальных параметров
    st.markdown(f"""
    <div style="display: flex; gap: 30px; margin-top: 20px; margin-bottom: 30px;">
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 20px 30px; width: 250px; background-color: #fcfcfc;">
            <div style="font-size: 16px; color: #666; margin-bottom: 10px;">Gradient Boosting</div>
            <div style="font-size: 42px; font-weight: bold; color: #2ca02c; line-height: 1.1;">
                {res['pred_gb']:.2f} <span style="font-size: 24px;">кг/м²</span>
            </div>
            <div style="font-size: 18px; color: gray; margin-top: 15px;">Оптимальный прогноз</div>
        </div>
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 20px 30px; width: 250px; background-color: #fcfcfc;">
            <div style="font-size: 16px; color: #666; margin-bottom: 10px;">Gaussian Process</div>
            <div style="font-size: 42px; font-weight: bold; color: #1f77b4; line-height: 1.1;">
                {res['pred_gp']:.2f} <span style="font-size: 24px;">кг/м²</span>
            </div>
            <div style="font-size: 16px; color: gray; margin-top: 15px;">
                ±1σ ({lower_bound:.2f} — {upper_bound:.2f})
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Рекомендуемые параметры среды")
    for f, v in res["params"].items():
        label = SLIDER_CONFIG.get(f, {"label": f})["label"]
        st.markdown(f"- {label}: **{v}**")
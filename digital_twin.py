import streamlit as st
import pandas as pd
import numpy as np
import joblib
import optuna

# Настройка страницы (широкий формат)
st.set_page_config(page_title="Цифровой двойник теплицы", layout="wide")

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

# 2. Конфигурация слайдеров
SLIDER_CONFIG = {
    "irrigation_illuminance": {"label": "Освещенность при поливе (Дж/см²)", "min": 100, "max": 2500, "step": 50, "default": 1500},
    "received_drainage": {"label": "Полученный дренаж (мл)", "min": 0, "max": 2000, "step": 50, "default": 500},
    "avg_outdoor_temp": {"label": "Ср. уличная темп. (°C)", "min": -15, "max": 35, "step": 0.5, "default": 15},
    "leaf_count": {"label": "Количество листьев (шт)", "min": 5, "max": 30, "step": 0.5, "default": 15},
    "substrate_EC": {"label": "Электропроводность (ЕС)", "min": 1.0, "max": 9.0, "step": 0.1, "default": 6.5},
    "consumed_volume": {"label": "Потребление воды (мл)", "min": 0, "max": 1500, "step": 10, "default": 500},
    "fruit_setting_cluster": {"label": "Завязавшиеся кисти (шт)", "min": 0, "max": 40, "step": 0.5, "default": 15},
    "total_illuminance": {"label": "Общая освещенность (Дж/см²)", "min": 0, "max": 3000, "step": 50, "default": 1500},
    "avg_night_temp": {"label": "Ср. ночная темп. (°C)", "min": 10, "max": 25, "step": 0.5, "default": 18},
    "leaf5_width": {"label": "Ширина 5-го листа (мм)", "min": 0, "max": 100, "step": 1, "default": 50},
}

# Загружаем модели
gp_model, gb_model, scaler_p, imp_p, feature_names = load_models()

if not feature_names:
    st.stop()

st.title("🌱 Цифровой двойник теплицы")
st.markdown("Настройте параметры среды, чтобы увидеть мгновенный прогноз урожайности.")

# 3. Отрисовка интерфейса
st.subheader("Параметры среды")
col1, col2 = st.columns(2)

for i, fname in enumerate(feature_names):
    config = SLIDER_CONFIG.get(fname, {"label": fname, "min": 0, "max": 100, "step": 1, "default": 10})

    # Инициализируем состояние для каждого ползунка, если его еще нет
    if fname not in st.session_state:
        st.session_state[fname] = float(config["default"])

    # Раскидываем по двум колонкам
    target_col = col1 if i < 5 else col2
    with target_col:
        st.slider(
            label=config["label"],
            min_value=float(config["min"]),
            max_value=float(config["max"]),
            step=float(config["step"]),
            key=fname  # Благодаря этому Streamlit слушает наше состояние
        )

# Собираем актуальные значения из состояния
inputs = {fname: st.session_state[fname] for fname in feature_names}

# 4. Мгновенный прогноз
df_input = pd.DataFrame([inputs]).reindex(columns=feature_names, fill_value=np.nan)
X_arr = imp_p.transform(df_input)
X_df = pd.DataFrame(X_arr, columns=feature_names)
X_scaled = scaler_p.transform(X_df)

pred_gb = float(gb_model.predict(X_df)[0])
pred_gp_arr, std_gp_arr = gp_model.predict(X_scaled, return_std=True)
pred_gp = float(pred_gp_arr[0])
std_val = float(std_gp_arr[0])

# Красивый вывод результатов
st.markdown(f"""
<div style="display: flex; gap: 30px; margin-top: 20px; margin-bottom: 30px;">
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 20px 30px; width: 250px; background-color: #fcfcfc;">
        <div style="font-size: 16px; color: #666; margin-bottom: 10px;">Gradient Boosting (GB)</div>
        <div style="font-size: 42px; font-weight: bold; color: #2ca02c; line-height: 1.1;">
            {pred_gb:.2f} <span style="font-size: 24px;">кг/м²</span>
        </div>
        <div style="font-size: 18px; color: gray; margin-top: 15px;">Точный прогноз</div>
    </div>
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 20px 30px; width: 250px; background-color: #fcfcfc;">
        <div style="font-size: 16px; color: #666; margin-bottom: 10px;">Gaussian Process (GP)</div>
        <div style="font-size: 42px; font-weight: bold; color: #1f77b4; line-height: 1.1;">
            {pred_gp:.2f} <span style="font-size: 24px;">кг/м²</span>
        </div>
        <div style="font-size: 18px; color: gray; margin-top: 15px;">
            ±{std_val:.2f} ({pred_gp - std_val:.2f} — {pred_gp + std_val:.2f})
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 5. Блок Байесовской оптимизации
st.divider()
st.subheader("Байесовская оптимизация (Optuna)")

# Инициализация памяти для хранения результатов
if "gb_results" not in st.session_state:
    st.session_state.gb_results = None
    st.session_state.gp_results = None
    st.session_state.opt_status = ""

# Кнопка запуска
if st.button("Найти оптимальные параметры"):
    st.session_state.opt_status = "⏳ Идет оптимизация... Пожалуйста, подождите (около 20 секунд)."
    st.rerun()

# Запуск процесса оптимизации
if st.session_state.opt_status.startswith("⏳"):
    with st.spinner("Optuna ищет лучшие комбинации..."):

        # Вспомогательная функция для подготовки данных
        def prepare_input_data(param_values):
            input_data = {name: val for name, val in zip(feature_names, param_values)}
            df_in = pd.DataFrame([input_data])
            df_aligned = df_in.reindex(columns=feature_names, fill_value=np.nan)
            arr_transformed = imp_p.transform(df_aligned)
            df_transformed = pd.DataFrame(arr_transformed, columns=feature_names)
            scaled_data = scaler_p.transform(df_transformed)
            return df_transformed, scaled_data

        # Целевые функции
        def objective_gb(trial):
            param_values = [trial.suggest_float(name, SLIDER_CONFIG[name]["min"], SLIDER_CONFIG[name]["max"]) for name in feature_names]
            df_transformed, _ = prepare_input_data(param_values)
            return float(gb_model.predict(df_transformed)[0])

        def objective_gp(trial):
            param_values = [trial.suggest_float(name, SLIDER_CONFIG[name]["min"], SLIDER_CONFIG[name]["max"]) for name in feature_names]
            _, scaled_data = prepare_input_data(param_values)
            pred_gp_opt, _ = gp_model.predict(scaled_data, return_std=True)
            return float(pred_gp_opt[0])

        # Запуск для GB
        sampler_gb = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42, n_startup_trials=50)
        study_gb = optuna.create_study(direction="maximize", sampler=sampler_gb)
        study_gb.optimize(objective_gb, n_trials=500)

        # Запуск для GP
        sampler_gp = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42, n_startup_trials=50)
        study_gp = optuna.create_study(direction="maximize", sampler=sampler_gp)
        study_gp.optimize(objective_gp, n_trials=500)

        # Вспомогательная функция для округления результатов
        def extract_params(study):
            params_dict = {}
            for name in feature_names:
                val = study.best_params[name]
                config = SLIDER_CONFIG.get(name, {"min": 0, "step": 1})
                step_str = str(config["step"])
                decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
                rounded_val = round((val - config["min"]) / config["step"]) * config["step"] + config["min"]
                params_dict[name] = round(rounded_val, decimals)
            return params_dict

        # Сохраняем результаты
        st.session_state.gb_results = {"params": extract_params(study_gb), "value": study_gb.best_value}
        st.session_state.gp_results = {"params": extract_params(study_gp), "value": study_gp.best_value}

        # --- АВТОМАТИЧЕСКОЕ ДВИЖЕНИЕ ПОЛЗУНКОВ ---
        # Берем лучшие параметры от CatBoost и записываем их в состояние слайдеров
        best_gb_params = extract_params(study_gb)
        for param_name, param_value in best_gb_params.items():
            st.session_state[param_name] = float(param_value)

        st.session_state.opt_status = "✅ Оптимизация успешно завершена! Ползунки обновлены."
        st.rerun()

# 6. Вывод результатов оптимизации
if st.session_state.opt_status:
    if st.session_state.opt_status.startswith("✅"):
        st.success(st.session_state.opt_status)
    else:
        st.info(st.session_state.opt_status)

if st.session_state.gb_results and st.session_state.gp_results:
    col_gb, col_gp = st.columns(2)

    with col_gb:
        st.markdown("### Gradient Boosting (GB)")
        st.markdown(f"<span style='color:#2ca02c; font-size:24px; font-weight:bold;'>{st.session_state.gb_results['value']:.2f} кг/м²</span>", unsafe_allow_html=True)
        st.divider()
        for feature, value in st.session_state.gb_results["params"].items():
            label = SLIDER_CONFIG.get(feature, {"label": feature})["label"]
            st.markdown(f"- {label}: **{value}**")

    with col_gp:
        st.markdown("### Gaussian Process (GP)")
        st.markdown(f"<span style='color:#1f77b4; font-size:24px; font-weight:bold;'>{st.session_state.gp_results['value']:.2f} кг/м²</span>", unsafe_allow_html=True)
        st.divider()
        for feature, value in st.session_state.gp_results["params"].items():
            label = SLIDER_CONFIG.get(feature, {"label": feature})["label"]
            st.markdown(f"- {label}: **{value}**")
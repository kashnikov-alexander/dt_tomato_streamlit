import streamlit as st
import pandas as pd
import numpy as np
import joblib
import optuna

# Настройка страницы (широкий формат)
st.set_page_config(page_title="Цифровой двойник теплицы", layout="wide")

# Отключаем лишние логи Оптуны
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Кэширование загрузки моделей (чтобы они не грузились каждый раз при клике)
@st.cache_resource
def load_models():
    try:
        gp = joblib.load("gp_model.pkl")
        gb = joblib.load("gb_model.pkl")
        scaler = joblib.load("scaler.pkl")
        imputer = joblib.load("imputer.pkl")
        features = scaler.feature_names_in_.tolist()
        return gp, gb, scaler, imputer, features
    except FileNotFoundError:
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
    st.error("❌ Ошибка: Файлы моделей (.pkl) не найдены. Убедитесь, что они лежат в папке с проектом.")
    st.stop()

st.title("🌱 Цифровой двойник теплицы")
st.markdown("Настройте параметры среды, чтобы увидеть мгновенный прогноз урожайности.")

# 3. Отрисовка интерфейса (2 колонки для слайдеров)
st.subheader("Параметры среды")
col1, col2 = st.columns(2)
inputs = {}

for i, fname in enumerate(feature_names):
    cfg = SLIDER_CONFIG.get(fname, {"label": fname, "min": 0, "max": 100, "step": 1, "default": 10})

    # Раскидываем по двум колонкам
    target_col = col1 if i < 5 else col2
    with target_col:
        inputs[fname] = st.slider(
            label=cfg["label"],
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(cfg["default"]),
            step=float(cfg["step"])
        )

# 4. Мгновенный прогноз (пересчитывается при любом движении ползунка)
df_input = pd.DataFrame([inputs]).reindex(columns=feature_names, fill_value=np.nan)
X_arr = imp_p.transform(df_input)
X_df = pd.DataFrame(X_arr, columns=feature_names)
X_scaled = scaler_p.transform(X_df)

pred_gb = float(gb_model.predict(X_df)[0])
pred_gp_arr, std_gp_arr = gp_model.predict(X_scaled, return_std=True)
pred_gp = float(pred_gp_arr[0])
std_val = float(std_gp_arr[0])

# Красивый вывод результатов через HTML
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
            ±{std_val:.2f} ({pred_gp - std_val:.2f} - {pred_gp + std_val:.2f})
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 5. Блок Байесовской оптимизации
st.divider()
st.subheader("Байесовская оптимизация (Optuna)")

# Инициализация памяти для хранения результатов Оптуны
if "gb_res" not in st.session_state:
    st.session_state.gb_res = None
    st.session_state.gp_res = None
    st.session_state.opt_status = ""

# Кнопка запуска
if st.button("Найти оптимальные параметры"):

    # Выводим сообщение о начале
    st.session_state.opt_status = "⏳ Идет оптимизация... Пожалуйста, подождите (около 20 секунд)."
    st.rerun() # Принудительно обновляем экран, чтобы показать текст

# Если нажата кнопка (и статус означает, что идет оптимизация)
if st.session_state.opt_status.startswith("⏳"):

    with st.spinner("Optuna ищет лучшие комбинации..."):
        # Функция для подготовки данных
        def _get_input_df(_params):
            _in_data = {_f: _v for _f, _v in zip(feature_names, _params)}
            _df_in = pd.DataFrame([_in_data])
            _df_al = _df_in.reindex(columns=feature_names, fill_value=np.nan)
            _X_arr = imp_p.transform(_df_al)
            _X_df = pd.DataFrame(_X_arr, columns=feature_names)
            _X_scaled = scaler_p.transform(_X_df)
            return _X_df, _X_scaled

        # Целевые функции
        def _objective_gb(_trial):
            _params = [_trial.suggest_float(_f, SLIDER_CONFIG[_f]["min"], SLIDER_CONFIG[_f]["max"]) for _f in feature_names]
            _X_df, _ = _get_input_df(_params)
            return float(gb_model.predict(_X_df)[0])

        def _objective_gp(_trial):
            _params = [_trial.suggest_float(_f, SLIDER_CONFIG[_f]["min"], SLIDER_CONFIG[_f]["max"]) for _f in feature_names]
            _, _X_scaled = _get_input_df(_params)
            _pred, _ = gp_model.predict(_X_scaled, return_std=True)
            return float(_pred[0])

        # Запуск для GB
        _sampler_gb = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42)
        _study_gb = optuna.create_study(direction="maximize", sampler=_sampler_gb)
        _study_gb.optimize(_objective_gb, n_trials=150)

        # Запуск для GP
        _sampler_gp = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42)
        _study_gp = optuna.create_study(direction="maximize", sampler=_sampler_gp)
        _study_gp.optimize(_objective_gp, n_trials=150)

        # Функция округления
        def _extract_params(_study):
            _dict = {}
            for _fname in feature_names:
                _val = _study.best_params[_fname]
                _cfg = SLIDER_CONFIG.get(_fname, {"min": 0, "step": 1})
                _step_str = str(_cfg["step"])
                _decimals = len(_step_str.split('.')[1]) if '.' in _step_str else 0
                _rounded_val = round((_val - _cfg["min"]) / _cfg["step"]) * _cfg["step"] + _cfg["min"]
                _dict[_fname] = round(_rounded_val, _decimals)
            return _dict

        # Сохраняем результаты в память Streamlit
        st.session_state.gb_res = {"params": _extract_params(_study_gb), "value": _study_gb.best_value}
        st.session_state.gp_res = {"params": _extract_params(_study_gp), "value": _study_gp.best_value}

        # Меняем статус
        st.session_state.opt_status = "✅ Оптимизация успешно завершена!"
        st.rerun()

# 6. Вывод результатов оптимизации
if st.session_state.opt_status:
    if st.session_state.opt_status.startswith("✅"):
        st.success(st.session_state.opt_status)
    else:
        st.info(st.session_state.opt_status)

if st.session_state.gb_res and st.session_state.gp_res:
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown("### Gradient Boosting (GB)")
        st.markdown(f"<span style='color:#2ca02c; font-size:24px; font-weight:bold;'>{st.session_state.gb_res['value']:.2f} кг/м²</span>", unsafe_allow_html=True)
        st.divider()
        for f, v in st.session_state.gb_res["params"].items():
            label = SLIDER_CONFIG.get(f, {"label": f})["label"]
            st.markdown(f"- {label}: **{v}**")

    with res_col2:
        st.markdown("### Gaussian Process (GP)")
        st.markdown(f"<span style='color:#1f77b4; font-size:24px; font-weight:bold;'>{st.session_state.gp_res['value']:.2f} кг/м²</span>", unsafe_allow_html=True)
        st.divider()
        for f, v in st.session_state.gp_res["params"].items():
            label = SLIDER_CONFIG.get(f, {"label": f})["label"]
            st.markdown(f"- {label}: **{v}**")
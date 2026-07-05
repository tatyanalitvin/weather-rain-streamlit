import joblib
import streamlit as st

from weather_data_processing import predict_input

MODEL_PATH = "aussie_rain.joblib"

st.set_page_config(
    page_title="Синоптична станція · Чи піде завтра дощ?",
    page_icon="🌧️",
    layout="wide",
)

# Стилі
# Палітра: синоптичний темно-синій (ink), холодний папір метеокарти,
# кобальт як сигнал дощу, вохра як сигнал сухої погоди.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

      :root {
        --ink:    #14243b;
        --paper:  #e9edf1;
        --cobalt: #2f4bd6;
        --ochre:  #e6982f;
        --euca:   #4f7a63;
        --line:   rgba(20,36,59,.14);
      }

      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      .stApp { background: var(--paper); }
      .block-container { max-width: 1080px; padding-top: 2.2rem; }

      /* Заголовки — грубий гротеск */
      h1, h2, h3, .hero h1 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -.01em; }

      /* ── Синоптична шапка з ізобарами ── */
      .hero {
        position: relative; overflow: hidden;
        background: var(--ink); color: #eef2f7;
        border-radius: 18px; padding: 2.4rem 2.2rem 2.1rem;
        margin-bottom: 1.6rem;
      }
      .hero .isobars { position: absolute; inset: 0; width: 100%; height: 100%;
        opacity: .22; }
      .hero .isobars path { fill: none; stroke: #7fa9c9; stroke-width: 1.4; }
      .hero-inner { position: relative; z-index: 1; }
      .hero h1 { margin: 0; font-size: 2.5rem; font-weight: 700; line-height: 1.05; }
      .hero h1 em { font-style: italic; color: #8fb6ff; font-weight: 600; }
      .hero p { margin: .8rem 0 0; max-width: 46ch; font-size: 1.02rem;
        color: #b9c6d6; line-height: 1.5; }

      /* Секційні підписи як на приладовій панелі */
      .section-label {
        font-family: 'IBM Plex Mono', monospace; font-size: .72rem;
        letter-spacing: .18em; text-transform: uppercase; color: var(--euca);
        margin: 1.4rem 0 .3rem; display: flex; align-items: center; gap: .6rem;
      }
      .section-label::after { content: ""; flex: 1; height: 1px; background: var(--line); }

      /* Числові показники — моноширинний «прилад» */
      [data-testid="stWidgetLabel"] p { font-size: .9rem; color: #33465e; }
      .stSlider [data-testid="stTickBarMin"],
      .stSlider [data-testid="stTickBarMax"] { font-family: 'IBM Plex Mono', monospace; }

      /* Вкладки */
      .stTabs [data-baseweb="tab-list"] { gap: .3rem; border-bottom: 1px solid var(--line); }
      .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif; font-weight: 500;
        color: #64748b; padding: .4rem .3rem;
      }
      .stTabs [aria-selected="true"] { color: var(--ink) !important; }

      /* Кнопка */
      div.stButton > button {
        background: var(--ink); color: #fff; border: 0; border-radius: 12px;
        padding: .8rem 1.4rem; width: 100%;
        font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.05rem;
        letter-spacing: .01em; transition: transform .12s ease, background .15s ease;
      }
      div.stButton > button:hover { background: var(--cobalt); transform: translateY(-1px); }
      div.stButton > button:focus-visible { outline: 3px solid var(--ochre); outline-offset: 2px; }

      /* ── Картка прогнозу з дилером-барометром ── */
      .forecast {
        display: flex; gap: 1.8rem; align-items: center; flex-wrap: wrap;
        background: #fff; border: 1px solid var(--line); border-left: 6px solid var(--accent, var(--cobalt));
        border-radius: 18px; padding: 1.7rem 1.9rem; margin-top: 1rem;
        box-shadow: 0 14px 34px rgba(20,36,59,.10);
        animation: rise .45s cubic-bezier(.2,.7,.3,1) both;
      }
      @keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }

      .dial { position: relative; width: 172px; height: 172px; flex: 0 0 auto; }
      .dial .ring {
        width: 100%; height: 100%; border-radius: 50%;
        background: conic-gradient(var(--accent) calc(var(--p) * 1%), rgba(20,36,59,.09) 0);
        -webkit-mask: radial-gradient(farthest-side, transparent 64%, #000 65%);
                mask: radial-gradient(farthest-side, transparent 64%, #000 65%);
      }
      .dial .face {
        position: absolute; inset: 0; display: flex; flex-direction: column;
        align-items: center; justify-content: center; text-align: center;
      }
      .dial .num { font-family: 'IBM Plex Mono', monospace; font-weight: 600;
        font-size: 2.35rem; color: var(--ink); line-height: 1; }
      .dial .num span { font-size: 1.1rem; color: #94a3b8; }
      .dial .cap { font-family: 'IBM Plex Mono', monospace; font-size: .62rem;
        letter-spacing: .14em; text-transform: uppercase; color: #94a3b8; margin-top: .35rem; }

      .verdict { flex: 1; min-width: 240px; }
      .verdict .tag {
        display: inline-block; font-family: 'IBM Plex Mono', monospace;
        font-size: .68rem; letter-spacing: .16em; text-transform: uppercase;
        color: var(--accent); border: 1px solid var(--accent); border-radius: 999px;
        padding: .2rem .7rem; margin-bottom: .6rem;
      }
      .verdict h2 { margin: 0; font-size: 1.85rem; color: var(--ink); }
      .verdict p { margin: .5rem 0 0; color: #506480; line-height: 1.5; }

      @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; transition: none !important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Синоптична шапка
st.markdown(
    """
    <div class="hero">
      <svg class="isobars" viewBox="0 0 1000 300" preserveAspectRatio="none">
        <path d="M-20 46 C 200 20, 340 74, 520 48 S 860 22, 1040 50"/>
        <path d="M-20 96 C 200 70, 340 124, 520 98 S 860 72, 1040 100"/>
        <path d="M-20 150 C 200 124, 340 178, 520 152 S 860 126, 1040 154"/>
        <path d="M-20 204 C 200 178, 340 232, 520 206 S 860 180, 1040 208"/>
        <path d="M-20 258 C 200 232, 340 286, 520 260 S 860 234, 1040 262"/>
      </svg>
      <div class="hero-inner">
        <h1>Чи піде <em>завтра</em> дощ?</h1>
        <p>Внесіть сьогоднішні спостереження за погодою — модель Random Forest
        оцінить ймовірність опадів на завтра для обраної метеостанції.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Завантажує модель разом з об'єктами препроцесингу (кешується)."""
    return joblib.load(MODEL_PATH)


model_dict = load_model()

# Опції полів беремо напряму з навчених препроцесорів, щоб вони
# збігалися з тим, на чому вчилась модель.
encoder = model_dict["encoder"]
categorical_cols = model_dict["categorical_cols"]
numeric_cols = model_dict["numeric_cols"]
cat_options = dict(zip(categorical_cols, encoder.categories_))

scaler = model_dict["scaler"]
imputer = model_dict["imputer"]
num_min = dict(zip(numeric_cols, scaler.data_min_))
num_max = dict(zip(numeric_cols, scaler.data_max_))
num_default = dict(zip(numeric_cols, imputer.statistics_))


def section(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def num_input(label, col, unit=""):
    """Числове поле з межами та значенням за замовчуванням із даних."""
    lo, hi = float(num_min[col]), float(num_max[col])
    return st.slider(
        f"{label}{unit}",
        min_value=round(lo, 1),
        max_value=round(hi, 1),
        value=round(float(num_default[col]), 1),
    )


def dir_input(label, col):
    opts = list(cat_options[col])
    return st.selectbox(label, opts, index=opts.index("W") if "W" in opts else 0)


# Журнал спостережень
locations = list(cat_options["Location"])
section("Локація та опади сьогодні")
c0, c1 = st.columns([2, 1])
with c0:
    location = st.selectbox(
        "Метеостанція", locations,
        index=locations.index("Sydney") if "Sydney" in locations else 0,
    )
with c1:
    rain_today = st.radio(
        "Чи йшов дощ сьогодні?", list(cat_options["RainToday"]), horizontal=True
    )

section("Сьогоднішні вимірювання")
tab_temp, tab_wind, tab_atmo, tab_cloud = st.tabs(
    ["🌡️ Температура", "💨 Вітер", "💧 Вологість і тиск", "☁️ Хмарність"]
)

with tab_temp:
    c1, c2, c3 = st.columns(3)
    with c1:
        min_temp = num_input("Мін. температура", "MinTemp", " °C")
        temp_9am = num_input("Температура о 9:00", "Temp9am", " °C")
    with c2:
        max_temp = num_input("Макс. температура", "MaxTemp", " °C")
        temp_3pm = num_input("Температура о 15:00", "Temp3pm", " °C")
    with c3:
        rainfall = num_input("Опади", "Rainfall", " мм")
        evaporation = num_input("Випаровування", "Evaporation", " мм")
        sunshine = num_input("Сонячне сяйво", "Sunshine", " год")

with tab_wind:
    c1, c2, c3 = st.columns(3)
    with c1:
        wind_gust_dir = dir_input("Напрямок пориву вітру", "WindGustDir")
        wind_gust_speed = num_input("Швидкість пориву", "WindGustSpeed", " км/год")
    with c2:
        wind_dir_9am = dir_input("Напрямок вітру о 9:00", "WindDir9am")
        wind_speed_9am = num_input("Швидкість вітру о 9:00", "WindSpeed9am", " км/год")
    with c3:
        wind_dir_3pm = dir_input("Напрямок вітру о 15:00", "WindDir3pm")
        wind_speed_3pm = num_input("Швидкість вітру о 15:00", "WindSpeed3pm", " км/год")

with tab_atmo:
    c1, c2 = st.columns(2)
    with c1:
        humidity_9am = num_input("Вологість о 9:00", "Humidity9am", " %")
        pressure_9am = num_input("Тиск о 9:00", "Pressure9am", " гПа")
    with c2:
        humidity_3pm = num_input("Вологість о 15:00", "Humidity3pm", " %")
        pressure_3pm = num_input("Тиск о 15:00", "Pressure3pm", " гПа")

with tab_cloud:
    c1, c2 = st.columns(2)
    with c1:
        cloud_9am = num_input("Хмарність о 9:00", "Cloud9am", " октантів")
    with c2:
        cloud_3pm = num_input("Хмарність о 15:00", "Cloud3pm", " октантів")

st.markdown("")
predict = st.button("Розрахувати прогноз на завтра  →")

# Прогноз
if predict:
    single_input = {
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evaporation,
        "Sunshine": sunshine,
        "WindGustDir": wind_gust_dir,
        "WindGustSpeed": wind_gust_speed,
        "WindDir9am": wind_dir_9am,
        "WindDir3pm": wind_dir_3pm,
        "WindSpeed9am": wind_speed_9am,
        "WindSpeed3pm": wind_speed_3pm,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm,
        "Cloud9am": cloud_9am,
        "Cloud3pm": cloud_3pm,
        "Temp9am": temp_9am,
        "Temp3pm": temp_3pm,
        "RainToday": rain_today,
    }

    prediction, probability = predict_input(model_dict, single_input)
    is_rain = str(prediction).lower() == "yes"

    # Ймовірність саме дощу (класу "Yes")
    rain_prob = probability if is_rain else 1 - probability

    accent = "#2f4bd6" if is_rain else "#e6982f"
    if is_rain:
        tag = "Прогноз: 🌧️ "
        title = "Завтра ймовірно піде дощ"
        note = (
            f"Модель оцінює ймовірність опадів у "
            f"<b>{location}</b> як <b>{rain_prob * 100:.0f}%</b>. "
            "Варто мати парасольку напоготові."
        )
    else:
        tag = "Прогноз: ☀️ "
        title = "Завтра дощу не очікується, сухо  "
        note = (
            f"Модель оцінює ймовірність опадів у "
            f"<b>{location}</b> лише як <b>{rain_prob * 100:.0f}%</b>. "
            "Ймовірно, день буде без дощу."
        )

    st.markdown(
        f"""
        <div class="forecast" style="--accent:{accent}; --p:{rain_prob * 100:.1f};">
          <div class="dial">
            <div class="ring"></div>
            <div class="face">
              <div class="num">{rain_prob * 100:.0f}<span>%</span></div>
              <div class="cap">ймовірність дощу</div>
            </div>
          </div>
          <div class="verdict">
            <span class="tag">{tag}</span>
            <h2>{title}</h2>
            <p>{note}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Що подавалося в модель (введені дані)"):
        st.json(single_input)

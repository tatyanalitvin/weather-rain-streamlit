# 🌧️ Rain in Australia: Streamlit + Random Forest

Веб-додаток на **Streamlit**, який прогнозує, чи піде **завтра дощ** в
Австралії, на основі сьогоднішніх погодних спостережень. Модель —
`RandomForestClassifier`, навчена на наборі даних
[weatherAUS](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)
(≈10 років щоденних спостережень Australian Bureau of Meteorology).

Домашнє завдання до теми **«Дерева прийняття рішень та випадкові ліси»**.

## Демо

>> https://weather-rain-app-ml.streamlit.app/

## Структура

| Файл | Призначення |
|------|-------------|
| `app.py` | Streamlit-додаток (інтерфейс + інференс) |
| `weather_data_processing.py` | Препроцесинг та функція `predict_input` |
| `train_model.py` | Навчання Random Forest і збереження моделі |
| `aussie_rain.joblib` | Навчена модель + препроцесори (стиснено, ~15 МБ) |
| `requirements.txt` | Залежності |

## Запуск локально

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Якість моделі

- Точність на тренувальній вибірці: **~0.91**
- Точність на валідаційній вибірці (2015 рік): **~0.86**

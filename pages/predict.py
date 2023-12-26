import streamlit as st
from lib import prediction_line, load_wineset
from models import load_model
import pandas as pd
from scipy import stats
import numpy as np

# prediction_line = prediction_line.reset_index()

buttons = {}

alpha = {
    "fixed acidity": "Нелетучие кислоты (г/л)",
    "volatile acidity": "Летучие кислоты (г/л)",
    "citric acid": "Лимонная кислота (г/л)",
    "residual sugar": "Остаточный сахар (г/л)",
    "chlorides": "Хлориды (г/л)",
    "free sulfur dioxide": "Свободный оксид серы (г/л)",
    "total sulfur dioxide": "Весь оксид серы (г/л)",
    "density": "Плотность (кг/л)",
    "pH": "Кислотность",
    "sulphates": "Сульфаты",
    "alcohol": "Содержание спирта (%)",
}

with st.sidebar:
    for column in load_wineset().drop("quality", axis = 1).columns:
        buttons[column] = st.number_input(f"{alpha[column]} ({load_wineset()[column].min() / 1.5 :.2f} - {load_wineset()[column].max() * 1.5 :.2f})", 
                                          min_value = load_wineset()[column].min() / 1.5,
                                          max_value = load_wineset()[column].max() * 1.5,
                                          value = load_wineset()[column].mean(),
                                          on_change = lambda: None,
                                          args = (),
                                          )

models = ["NaiveBayes", "GradBoost", "Bagging", "Stacking"]

st.title("Предсказания моделей")
X = np.fromiter((value for value in buttons.values()), float).reshape(1, -1)
y = []

for model in models:
    mdl = load_model(model)
    prediction = mdl.predict(X)
    y += [prediction]
    st.subheader(f"{model}: {int(prediction)}")

st.header(f"Финальный результат: {int(stats.mode(y).mode)}")

# st.data_editor
# https://docs.streamlit.io/library/api-reference/data/st.data_editor
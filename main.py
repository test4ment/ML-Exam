import streamlit as st
import pandas as pd
import numpy as np
import pickle

from st_pages import Page, Section, show_pages, add_page_title

# st.set_page_config(page_title = "New Name")

show_pages(
    [
        Page("main.py", "РГР"),
        Page("pages/about.py", "Об авторе"),
        Page("pages/about_dataset.py", "О датасете"),
        Page("pages/visualize.py", "Визуализация"),
        Page("pages/predict.py", "Предсказание модели"),
    ]
)

st.title('Расчетно-графическая работа')

st.text('Тема: инференс моделей машинного обучения')

import sklearn.gaussian_process
import streamlit as st

from st_pages import Page, Section, show_pages, add_page_title

# st.set_page_config(page_title = "New Name")

show_pages(
    [
        Page("pages/predict.py", "Предсказание модели"),
    ]
)

st.title("Расчетно-графическая работа")

st.text("Тема: инференс (вывод) машинного обучения")
st.text("Выполнил: студент группы ФИТ-221 Калицкий Никита Николаевич")

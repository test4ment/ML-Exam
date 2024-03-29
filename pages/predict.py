import streamlit as st
import pandas as pd
import numpy as np
import pickle

# prediction_line = prediction_line.reset_index()

df = pd.read_csv("DataExam_proc_smaller.csv")

buttons = {}

with st.sidebar:
    for column in df.drop("Revenue", axis = 1).columns:
        buttons[column] = st.number_input(f"{column} ({df[column].min() / 1.5 :.2f} - {df[column].max() * 1.5 :.2f})", 
                                          min_value = df[column].min() / 1.5,
                                          max_value = df[column].max() * 1.5,
                                          value = df[column].mean(),
                                          on_change = lambda: None,
                                          args = (),
                                          )

st.title("Предсказание модели")
X = np.array([np.array([value for value in buttons.values()])])
st.write(X)
y = []

with open("rforest.model", "rb") as f:
    mdl = pickle.load(f)
prediction = mdl.predict(X)
st.header(f"{prediction}")

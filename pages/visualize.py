import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/winetask.csv")

fig, ax = plt.subplots()
ax.boxplot(df)

st.pyplot(fig)
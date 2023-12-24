import pandas as pd
import streamlit as st

def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
  Q1 = data.quantile(0.25)
  Q3 = data.quantile(0.75)
  IQR = Q3 - Q1

  return data[(data >= Q1 - 1.5 * IQR) & (data <= Q3 + 1.5 * IQR)]

@st.cache_data
def load_wineset() -> pd.DataFrame:
  return pd.read_csv("/datasets/winetask.csv")
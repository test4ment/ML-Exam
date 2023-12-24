import pandas as pd
import streamlit as st

def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
  Q1 = data.quantile(0.25)
  Q3 = data.quantile(0.75)
  IQR = Q3 - Q1

  return data[(data >= Q1 - 1.5 * IQR) & (data <= Q3 + 1.5 * IQR)]

@st.cache_data
def load_wineset() -> pd.DataFrame:
  return pd.read_csv("datasets\winetask.csv")

global tree_feature_importances
tree_feature_importances = [0.06582553, 0.13270819, 0.06301554, 0.05962406, 0.07057525, 0.08254963, 0.08072115, 0.0733181, 0.0564277, 0.07822854, 0.23700631]
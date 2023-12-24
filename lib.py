import pandas as pd

def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
  Q1 = data.quantile(0.25)
  Q3 = data.quantile(0.75)
  IQR = Q3 - Q1

  return data[(data >= Q1 - 1.5 * IQR) & (data <= Q3 + 1.5 * IQR)]
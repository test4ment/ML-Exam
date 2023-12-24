import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lib import load_wineset, tree_feature_importances
import numpy as np

df = load_wineset()

subsets = [['free sulfur dioxide', 'total sulfur dioxide'], 
           ['fixed acidity'], 
           ['residual sugar'], 
           ['chlorides'],
           ['sulphates'], 
           ['density'],
           ['pH'],
           ['alcohol'], 
           ['quality']]

figs_axs = []
figs_axs += plt.subplots(len(subsets) // 2 + 1, 1, figsize = (3.5, 10))
figs_axs += plt.subplots(len(subsets) // 2, 1, figsize = (3.5, 10.125))

subsets = [
    subsets[:len(subsets)//2 + 1],
    subsets[len(subsets)//2 + 1:]
]

st.title("Анализ выбросов")

colns = list(st.columns(2))

for col in range(len(colns)):
    for ax, subset in zip(figs_axs[col * 2 + 1], subsets[col]):
        
        ax.boxplot(df[subset], labels = subset)

    colns[col].pyplot(figs_axs[col * 2])

st.title("Значимость признаков")


"Дисперсия:"
stds = df.describe().loc[["std"]].drop("quality", axis = 1)
fig, ax = plt.subplots(figsize = (4, 2))
ax.barh(stds.columns, stds.to_numpy()[0])
st.pyplot(fig)

"Значимость признаков в деревье решений:"
fig, ax = plt.subplots(figsize = (4, 2))

tree_feature_importances, columns = zip(*sorted(zip(tree_feature_importances, df.drop("quality", axis = 1).columns), key = lambda x: x[0]))
ax.barh(width = tree_feature_importances, y = columns)
st.pyplot(fig)


st.title("Зависимость средних значений признаков и качества")
df_means = df.groupby("quality").aggregate(np.mean).T
fig, axs = plt.subplots(7, layout='constrained', figsize = (4, 16))

for num, ax in enumerate(axs):
    ax.plot(df_means.iloc[num - 1], color = "y")
    ax.set_title((df_means.iloc[num - 1]).name)
st.pyplot(fig)

st.title("Распределения")

for column in df.columns:
    fig, ax = plt.subplots(figsize = (6, 3))
    
    n, bins, patches = ax.hist(df[column], bins = 29, range = (df[column].quantile(0.1), df[column].quantile(0.9)), color = "tab:red", density=True)

    mu = df[column].mean()
    sigma = df[column].std()

    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

    ax.plot(bins, y, '--')

    ax.set_title(column)
    st.pyplot(fig)


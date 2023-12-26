import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lib import load_wineset, tree_feature_importances
import numpy as np
from models import load_model
from sklearn.metrics import accuracy_score, classification_report

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
df = df.drop("quality", axis = 1)

for column in df.columns:
    fig, ax = plt.subplots(figsize = (6, 3))
    
    n, bins, patches = ax.hist(df[column], bins = 29, range = (df[column].quantile(0.1), df[column].quantile(0.9)), color = "tab:red", density=True)

    mu = df[column].mean()
    sigma = df[column].std()

    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

    ax.plot(bins, y, '--')

    ax.set_title(column)
    st.pyplot(fig)


st.title("Точность моделей")

df = load_wineset()
df = df[(df["quality"] >= 4) & (df["quality"] <= 7)]
df = df.sort_values("quality")
df_X = df.drop("quality", axis = 1)
df_y = df["quality"]

models = ["NaiveBayes", "GradBoost", "Bagging", "Stacking", 
        #   "DNN"
          ]

st.subheader(f"{models[0]}")
md = load_model(models[0])
report_df = pd.DataFrame(classification_report(df_y, md.predict(df_X), output_dict = True))
st.dataframe(report_df)


fig, ax = plt.subplots(figsize = (10, 10))
# bool_arr = (df_y == md.predict(df_X)).to_numpy()

# df_y = df_y.reset_index()
st.write(df_y)
bounds = {}

for num, i in enumerate(np.unique(df_y), start = 4):
    sub = np.argwhere(df_y.to_numpy() == i)
    bounds[num] = (int(sub[0]), int(sub[-1]))

for num2, model in enumerate(models):
    md = load_model(model)
    bool_arr = (df_y == md.predict(df_X)).to_numpy()
    st.write(sum(bool_arr) / len(bool_arr))
    for check, i in enumerate(bounds.values()):
        color = "black" if check % 2 == 0 else "green"
        ax.eventplot((np.argwhere(bool_arr[i[0] : (i[1] - 1)]).T + i[0]), linewidth = 0.1, color = color, linelength = 0.33, lineoffsets = num2)
    # ax.eventplot((np.argwhere(bool_arr[i[0] : (i[1] - 1)]).T + i[0]), linewidth = 0.1, color = color, linelength = 0.33)

ax.set_ylabel("bayes")
ax.set_xlim((0, df.shape[0]))
ax.get_yaxis().set_ticks([])
# ax.get_xaxis().set_ticks([4, 5, 6, 7])
st.pyplot(fig)
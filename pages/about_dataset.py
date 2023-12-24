from lib import load_wineset

st.title("О датасете")

st.text("В датасете собраны химические и физические величины разных сортов вин, выжатых из винограда из примерно одной области, а также оценка дегустации")
st.text("Перечислены следующие величины:")

st.text("fixed acidicty", help = "Нелетучие кислоты (г/л)")
st.text("volatile acidicty", help = "Летучие кислоты (г/л)")
st.text("citric acid", help = "Лимонная кислота (г/л)")
st.text("residual sugar", help = "Остаточный сахар (г/л)")
st.text("chlorides", help = "Хлориды (г/л)")
st.text("free sulfur dioxide", help = "Свободные оксиды серы (г/л)")
st.text("total sulfur dioxide", help = "Все оксиды серы (г/л)")
st.text("density", help = "Плотность (кг/л)")
st.text("pH", help = "Водородный показатель кислотности")
st.text("sulphates", help = "Сульфаты (г/л)")
st.text("alcohol", help = "Содержание спирта (%)")

st.text("В колонке quality указана дегустационная оценка")

st.dataframe(load_wineset())

st.dataframe(load_wineset().describe())
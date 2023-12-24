import tensorflow as tf
import pickle

@st.cache_resource
def load_model(str: model):
    path = "models/"
    {
        "NaiveBayes": lambda: pickle.load(open(path + "GaussianNB.model")),
        "RFE_3": lambda: pickle.load(open(path + "RFE_3f.model")),
        "RFE_6": lambda: pickle.load(open(path + "RFE_6f.model")),
        "GradBoost": lambda: pickle.load(open(path + "GradientBoostingClassifier_v2.model")),
        "Bagging": lambda: pickle.load(open(path + "Bagging_Tree_n23.model")),
        "Stacking": lambda: pickle.load(open(path + "StackingClassifier_kNN_Tree_LDA.model")),
        "DNN": lambda: tf.keras.models.load_model(path + "tf_model_5.keras"),
    }[model]()
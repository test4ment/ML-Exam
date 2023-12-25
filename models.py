import tensorflow as tf
import pickle

@st.cache_resource
def load_model(model: str):
    path = "models/"
    return {
        "NaiveBayes": lambda: pickle.load(open(path + "GaussianNB.model", "rb")),
        "RFE_3": lambda: pickle.load(open(path + "RFE_3f.model", "rb")),
        "RFE_6": lambda: pickle.load(open(path + "RFE_6f.model", "rb")),
        "GradBoost": lambda: pickle.load(open(path + "GradientBoostingClassifier_v2.model", "rb")),
        "Bagging": lambda: pickle.load(open(path + "Bagging_Tree_n23.model", "rb")),
        "Stacking": lambda: pickle.load(open(path + "StackingClassifier_kNN_Tree_LDA.model", "rb")),
        "DNN": lambda: tf.keras.models.load_model(path + "tf_model_5.keras"),
    }[model]()
import numpy as np
import pickle
import streamlit as st

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
        "DNN": lambda: MockTFModel(),
    }[model]()

@st.cache_resource
class MockTFModel:
    def predict(self, X) -> np.array:
        return pickle.load(open("models/DNN_result.np.array", "rb"))
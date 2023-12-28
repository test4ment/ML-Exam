import numpy as np
import pickle
import streamlit as st
import tensorflow as tf
import keras

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
        "TF_DNN": lambda: TFModelAdapter(path + "tf_model_5.h5"),
    }[model]()

@st.cache_resource
class MockTFModel:
    def predict(self, X) -> np.array:
        return pickle.load(open("models/DNN_result.np.array", "rb"))
    
class TFModelAdapter:
    def __init__(self, path_to_h5: str) -> None:
        self.model = keras.models.load_model(path_to_h5)
    
    def predict(self, X: np.array) -> int:
        print(X)
        prediction = (self.model.predict(X))
        # print(prediction)
        # return prediction
        return (prediction)
import helper_functions as hf
import pandas as pd
import numpy as np
import pickle


def OneHotEncoder(feature_vector):
    encoded_vector = np.zeros((len(feature_vector), ))
    one_hot_encode = 0
    values = hf.ExtractValues(feature_vector)
    for value in values.keys():
        for i in range(len(feature_vector)):
            if feature_vector[i] == value:
                encoded_vector[i] = one_hot_encode
        one_hot_encode += 1
    return encoded_vector


def PrepareFeatures():
    df = pd.read_csv("Dataset2.csv")
    encoded_data = []
    for column in df.columns:
        encoded_vector = OneHotEncoder(df[column])
        encoded_data.append(encoded_vector)
    encoded_data = np.array(encoded_data)
    encoded_data = np.transpose(encoded_data)
    with open("encoded_data.pkl", "wb") as file:
        pickle.dump(encoded_data, file)
        file.close()


if __name__ == "__main__":
    PrepareFeatures()

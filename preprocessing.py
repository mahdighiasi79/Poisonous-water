import helper_functions as hf
import pandas as pd
import numpy as np
import pickle


def OneHotEncoder(feature_vector):
    dictionary = hf.ExtractValues(feature_vector)
    values = dictionary.keys()
    encoded_vector = np.zeros((len(feature_vector), len(values)))
    i = 0
    for value in values:
        for j in range(len(feature_vector)):
            if feature_vector[j] == value:
                encoded_vector[j][i] = 1
        i += 1
    return encoded_vector


def PrepareFeatures():
    df = pd.read_csv("Dataset2.csv")
    encoded_data = np.array([[]] * len(df))
    for column in df.columns:
        encoded_vector = OneHotEncoder(df[column])
        encoded_data = np.append(encoded_data, encoded_vector, axis=1)
    encoded_data = np.delete(encoded_data, [0, 1], axis=1)

    with open("features.pkl", "wb") as file:
        pickle.dump(encoded_data, file)
        file.close()

    poisonous = np.array(df["poisonous"])
    labels = (poisonous == 'e').astype(int)

    with open("labels.pkl", "wb") as file:
        pickle.dump(labels, file)
        file.close()


if __name__ == "__main__":
    PrepareFeatures()

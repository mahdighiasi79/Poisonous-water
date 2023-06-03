import math
import pickle

import numpy as np

train_set = []

# hyperparameters
k = 8


def EuclideanDistance(point):
    distance = np.power(point - train_set, 2)
    distance = np.sum(distance, axis=1, keepdims=False)
    distance = np.power(distance, 0.5)
    return distance


def Predict(point):
    distances = EuclideanDistance(point)
    indexes = np.arange(len(train_set))
    hash_table = dict(zip(distances, indexes))
    sorted_distances = sorted(distances, reverse=True)

    p = 0
    e = 0
    for i in range(k):
        distance = sorted_distances[i]
        record_id = hash_table[distance]
        if train_set[record_id][0] == 0:
            p += 1
        else:
            e += 1

    if e > p:
        return 1
    else:
        return 0


if __name__ == "__main__":
    with open("encoded_data.pkl", "rb") as file:
        data = pickle.load(file)
        file.close()

    train_set = data[:math.floor(0.8 * len(data))]
    cross_validation_set = data[math.floor(0.8 * len(data)):math.floor(0.9 * len(data))]
    test_set = data[math.floor(0.9 * len(data)):]

    true_predictions = 0
    for record in cross_validation_set:
        label = record[0]
        predicted_label = Predict(record)
        if label == predicted_label:
            true_predictions += 1
    accuracy = (true_predictions / len(cross_validation_set)) * 100
    print(accuracy)
import math
import pickle
import numpy as np

train_set = []


def EuclideanDistance(point):
    distance = np.power(train_set - point, 2)
    distance = np.sum(distance, axis=1, keepdims=False)
    distance = np.power(distance, 0.5)
    return distance


def Predict(point, k):
    distances = EuclideanDistance(point)
    indexes = np.arange(len(train_set))
    hash_table = dict(zip(distances, indexes))
    sorted_distances = sorted(distances)

    p = 0
    e = 0
    for nearest_neighbors in range(k):
        distance = sorted_distances[nearest_neighbors]
        record_id = hash_table[distance]
        if train_labels[record_id] == 0:
            p += 1
        else:
            e += 1

    if e > p:
        return 1
    else:
        return 0


if __name__ == "__main__":
    with open("features.pkl", "rb") as file:
        features = pickle.load(file)
        file.close()
    with open("labels.pkl", "rb") as file:
        labels = pickle.load(file)
        file.close()
    num_records = len(features)
    test_set_size = math.floor(0.1 * num_records)

    average_accuracy = 0
    for i in range(9):
        train_set = np.append(features[:i * test_set_size], features[(i + 2) * test_set_size:], axis=0)
        train_labels = np.append(labels[:i * test_set_size], labels[(i + 2) * test_set_size:], axis=0)
        cross_validation_set = features[i * test_set_size: (i + 1) * test_set_size]
        cross_validation_labels = labels[i * test_set_size: (i + 1) * test_set_size]
        test_set = features[(i + 1) * test_set_size: (i + 2) * test_set_size]
        test_labels = labels[(i + 1) * test_set_size: (i + 2) * test_set_size]

        best_accuracy = -np.inf
        flag = True
        K = 0
        while flag:
            true_predictions = 0
            for j in range(test_set_size):
                label = cross_validation_labels[j]
                predicted_label = Predict(cross_validation_set[j], K)
                if label == predicted_label:
                    true_predictions += 1
            cross_accuracy = (true_predictions / test_set_size) * 100
            print("round:", i)
            print("k:", K)
            print("accuracy:", cross_accuracy)
            print("/////////////////////////////////////////////////////////")

            if cross_accuracy > best_accuracy:
                best_accuracy = cross_accuracy
                K += 1
            else:
                flag = False
                K -= 1

        true_predictions = 0
        for j in range(test_set_size):
            label = test_labels[j]
            predicted_label = Predict(test_set[j], K)
            if label == predicted_label:
                true_predictions += 1
        accuracy = (true_predictions / test_set_size) * 100
        print("end of round:", i)
        print("k:", K)
        print("accuracy:", accuracy)
        print("////////////////////////////////////////////////////////////////////")

        average_accuracy += accuracy

    average_accuracy /= 9
    print("average accuracy:", average_accuracy)

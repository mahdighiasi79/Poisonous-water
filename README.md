# Poisonous-water
In this project, we aim to predict whether a given area's water is poisonous or not.
The features that we use for this task are "cap-color", "cap-shape", "cap-surface", "population", "habitat", etc.
In the "poisonous" column, it's indicated whether the water is poisonous or not. 'p' means poisonous and 'e' means not poisonous.
So, there are two classes of records.

At the preprocessing step, we convert all categorical features into numerical ones by one hot encoding.
Before one hot encoding, we had 22 features but after performing one hot encoding, we will have 27 features.

The model we used for this classification tgask is K Nearest Neighbors (KNN). 
The euclidean distance is used as the distance metric between points.

10-fold-cross-validation testing has been executed for evaluating our model's performance.
Based on our evaluation, KNN gives an averege accuracy of about 99.74%.

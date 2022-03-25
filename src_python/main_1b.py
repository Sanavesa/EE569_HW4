import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV

label_lookup = {"blanket": 0, "brick": 1, "grass": 2, "stones": 3}
inv_label_lookup = {v:k for k,v in label_lookup.items()}
color_lookup = {0: "blue", 1: "red", 2: "green", 3: "black"}

# Load Train Feature Vectors
train_x = np.zeros((36, 25))
train_y = [0] * 36 # index of label (0=blanket, ...)
train_labels = [""] * 36 # the label (word) of each instance (i.e. blanket)
with open("build/debug/train_features.csv", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        vals = line.split(",")[:-1]
        train_x[i, :] = vals[1:]
        train_y[i] = label_lookup[vals[0]]
        train_labels[i] = vals[0]
        
# Load Test Feature Vectors    
test_x = np.zeros((12, 25))
test_y = [0] * 12 # index of label (0=blanket, ...)
test_labels = ["grass", "blanket", "blanket", "stones", "stones", "grass", "brick", "stones", "brick", "brick", "blanket", "grass"]
with open("build/debug/test_features.csv", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        vals = line.split(",")[:-1]
        test_x[i, :] = vals[1:]
        test_y[i] = label_lookup[test_labels[i]]

# Normalize the 25D
scaler = MinMaxScaler()
scaler.fit(train_x)
train_x_scaled = scaler.transform(train_x)
test_x_scaled = scaler.transform(test_x)

# PCA
pca = PCA(n_components=3)
pca.fit(train_x)
train_x_pca = pca.transform(train_x)
test_x_pca = pca.transform(test_x)

pca_scaled = PCA(n_components=3)
pca_scaled.fit(train_x_scaled)
train_x_pca_scaled = pca_scaled.transform(train_x_scaled)
test_x_pca_scaled = pca_scaled.transform(test_x_scaled)

# returns the accuracy%, correct matching, total samples
def get_accuracy(pred, true):
    matching_classifications = len([i for i, j in zip(pred, true) if i == j])
    accuracy = matching_classifications / len(true)
    return accuracy, matching_classifications

########################## K-Means on 25D ##########################
print("-"*50, "Kmeans on 25D", "-"*50)
kmeans25 = KMeans(n_clusters=4, random_state=2150, init="k-means++").fit(train_x_scaled)

# Majority voting on the clusters using the training dataset
kmeans25_train_prediction = kmeans25.predict(train_x_scaled)
kmeans25_clusterlabel = ["NONE"] * 4
for clusterIdx in range(4):
    votes = np.zeros(4) # vote count for each class in this cluster
    for i, pred in enumerate(kmeans25_train_prediction):
        if pred == clusterIdx:
            predictedClass = train_y[i]
            votes[predictedClass] += 1
    # skip empty clusters
    if votes.sum() == 0:
        continue
    clusterClass = votes.argmax()
    clusterLabel = inv_label_lookup[clusterClass]
    clusterPurity = votes.max() / votes.sum()
    kmeans25_clusterlabel[clusterIdx] = clusterLabel
    print("cluster", clusterIdx, "votes", votes, "class", clusterClass, "label", clusterLabel, "purity", clusterPurity)

# Label the testing dataset using the kmeans clusters
kmeans25_test_prediction = kmeans25.predict(test_x_scaled)
kmeans25_classification = [kmeans25_clusterlabel[i] for i in kmeans25_test_prediction]
print("Kmeans25", kmeans25_classification)

# Compute testing accuracy and error rate
accuracy, matching_classifications = get_accuracy(kmeans25_classification, test_labels)
print(f"Accuracy: {matching_classifications} / {len(test_labels)} = {accuracy}")
print(f"Error Rate: {len(test_labels) - matching_classifications} / {len(test_labels)} = {1.0 - accuracy}")
print()


########################## K-Means on 3D ##########################
print("-"*50, "Kmeans on 3D", "-"*50)
kmeans3 = KMeans(n_clusters=4, random_state=12809, init="k-means++").fit(train_x_pca_scaled)

# Majority voting on the clusters using the training dataset
kmeans3_train_prediction = kmeans3.predict(train_x_pca_scaled)
kmeans3_clusterlabel = ["NONE"] * 4
for clusterIdx in range(4):
    votes = np.zeros(4) # vote count for each class in this cluster
    for i, pred in enumerate(kmeans3_train_prediction):
        if pred == clusterIdx:
            predictedClass = train_y[i]
            votes[predictedClass] += 1
    # skip empty clusters
    if votes.sum() == 0:
        continue
    clusterClass = votes.argmax()
    clusterLabel = inv_label_lookup[clusterClass]
    clusterPurity = votes.max() / votes.sum()
    kmeans3_clusterlabel[clusterIdx] = clusterLabel
    print("cluster", clusterIdx, "votes", votes, "class", clusterClass, "label", clusterLabel, "purity", clusterPurity)

# Label the testing dataset using the kmeans clusters
kmeans3_test_prediction = kmeans3.predict(test_x_pca_scaled)
kmeans3_classification = [kmeans3_clusterlabel[i] for i in kmeans3_test_prediction]
print("Kmeans3", kmeans3_classification)

# Compute testing accuracy and error rate
accuracy, matching_classifications = get_accuracy(kmeans3_classification, test_labels)
print(f"Accuracy: {matching_classifications} / {len(test_labels)} = {accuracy}")
print(f"Error Rate: {len(test_labels) - matching_classifications} / {len(test_labels)} = {1.0 - accuracy}")
print()


########################## Random Forest on 3D ##########################
print("-"*50, "Random Forest on 3D", "-"*50)

rf = RandomForestClassifier(max_depth=3, random_state=13)
rf.fit(train_x_pca, train_y)

rf_train_pred = rf.predict(train_x_pca)
accuracy, matching_classifications = get_accuracy(rf_train_pred, train_y)
print(f"RF Train Accuracy: {matching_classifications} / {len(train_y)} = {accuracy}")
print("RF Train Predictions: ", rf_train_pred)

rf_test_pred = rf.predict(test_x_pca)
accuracy, matching_classifications = get_accuracy(rf_test_pred, test_y)
print(f"RF Test Accuracy: {matching_classifications} / {len(test_y)} = {accuracy}")
print("RF Test Predictions: ", rf_test_pred)
print("RF Classifications", [inv_label_lookup[i] for i in rf_test_pred])
print()

########################## SVM on 3D ##########################
print("-"*50, "SVM on 3D", "-"*50)

svm = SVC(random_state=1337, C=100)
svm.fit(train_x_pca, train_y)

svm_train_pred = svm.predict(train_x_pca)
accuracy, matching_classifications = get_accuracy(svm_train_pred, train_y)
print(f"SVM Train Accuracy: {matching_classifications} / {len(train_y)} = {accuracy}")
print("SVM Train Predictions: ", svm_train_pred)

svm_test_pred = svm.predict(test_x_pca)
accuracy, matching_classifications = get_accuracy(svm_test_pred, test_y)
print(f"SVM Test Accuracy: {matching_classifications} / {len(test_y)} = {accuracy}")
print("SVM Test Predictions: ", svm_test_pred)
print("SVM Classifications", [inv_label_lookup[i] for i in svm_test_pred])
print()
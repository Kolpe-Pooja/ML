import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('K-Means_Clustering.csv')

# Check column names
print(data.columns)
X = data[["CREDIT_LIMIT", "PURCHASES"]].dropna()
X = X.sample(100)  # Limit to 100 rows to test

# Choose columns that exist
X = data[["PURCHASES", "CREDIT_LIMIT"]].dropna()

# Visualize data points
plt.scatter(X["CREDIT_LIMIT"], X["PURCHASES"], c='black')
plt.xlabel('Credit Limit')
plt.ylabel('Purchases')
plt.show()

# Choose number of clusters
K = 3

# Randomly choose centroids
Centroids = X.sample(n=K)
plt.scatter(X["CREDIT_LIMIT"], X["PURCHASES"], c='black')
plt.scatter(Centroids["CREDIT_LIMIT"], Centroids["PURCHASES"], c='red')
plt.xlabel('Credit Limit')
plt.ylabel('Purchases')
plt.show()

# K-Means Clustering
diff = 1
j = 0

while diff != 0:
    XD = X.copy()
    i = 1
    for index1, row_c in Centroids.iterrows():
        ED = []
        for index2, row_d in XD.iterrows():
            d1 = (row_c["CREDIT_LIMIT"] - row_d["CREDIT_LIMIT"]) ** 2
            d2 = (row_c["PURCHASES"] - row_d["PURCHASES"]) ** 2
            d = np.sqrt(d1 + d2)
            ED.append(d)
        X[i] = ED
        i += 1

    C = []
    for index, row in X.iterrows():
        min_dist = row[1]
        pos = 1
        for i in range(K):
            if row[i + 1] < min_dist:
                min_dist = row[i + 1]
                pos = i + 1
        C.append(pos)
    X["Cluster"] = C
    Centroids_new = X.groupby(["Cluster"]).mean()[["PURCHASES", "CREDIT_LIMIT"]]
    if j == 0:
        diff = 1
        j += 1
    else:
        diff = (Centroids_new["PURCHASES"] - Centroids["PURCHASES"]).sum() + (Centroids_new["CREDIT_LIMIT"] - Centroids["CREDIT_LIMIT"]).sum()
        print(diff.sum())
    Centroids = Centroids_new

# Plot clusters
color = ['blue', 'green', 'cyan']
for k in range(K):
    data_k = X[X["Cluster"] == k + 1]
    plt.scatter(data_k["CREDIT_LIMIT"], data_k["PURCHASES"], c=color[k])
plt.scatter(Centroids["CREDIT_LIMIT"], Centroids["PURCHASES"], c='red')
plt.xlabel('Credit Limit')
plt.ylabel('Purchases')
plt.show()

import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Load original UCI Mushroom dataset
file_path = "../01_Dataset/mushroom/agaricus-lepiota.data"

columns = [
    "class", "cap-shape", "cap-surface", "cap-color",
    "bruises", "odor", "gill-attachment", "gill-spacing",
    "gill-size", "gill-color", "stalk-shape", "stalk-root",
    "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring",
    "veil-type", "veil-color", "ring-number", "ring-type",
    "spore-print-color", "population", "habitat"
]

df = pd.read_csv(file_path, header=None, names=columns)

# Exclude target class to prevent data leakage
X = df.drop(columns=["class"]).replace("?", "missing")

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown="ignore")
X_encoded = encoder.fit_transform(X)

print("Number of records:", X.shape[0])
print("Encoded feature count:", X_encoded.shape[1])

# Run DBSCAN
start = time.time()

model = DBSCAN(
    eps=0.15,
    min_samples=10,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

labels = model.fit_predict(X_encoded)

execution_time = time.time() - start

# Count clusters and noise
unique_labels = set(labels)
cluster_labels = unique_labels - {-1}

print("\nDBSCAN Results")
print("Number of clusters:", len(cluster_labels))
print("Number of noise records:", list(labels).count(-1))
print("Execution time:", round(execution_time, 3), "seconds")

for cluster in sorted(cluster_labels):
    print(
        "Cluster", cluster,
        "size:", list(labels).count(cluster)
    )

# Save cluster assignments
output = df.copy()
output["DBSCAN_cluster"] = labels
output.to_csv("DBSCAN_Cluster_Results.csv", index=False)

# Silhouette score, only when at least two clusters exist
if len(cluster_labels) >= 2:
    non_noise = labels != -1
    if len(set(labels[non_noise])) >= 2:
        score = silhouette_score(
            X_encoded[non_noise],
            labels[non_noise],
            metric="cosine"
        )
        print("Silhouette score (excluding noise):", round(score, 4))
else:
    print("Silhouette score unavailable: fewer than two clusters.")

print("\nDBSCAN completed successfully!")

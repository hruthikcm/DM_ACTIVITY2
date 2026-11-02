
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA

# Create output folder
os.makedirs("EDA_Output", exist_ok=True)

# Load original dataset
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

print("DATASET OVERVIEW")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nFirst five records:")
print(df.head())

# 1. Missing-value analysis
missing = df.replace("?", np.nan).isnull().sum()
missing.to_csv("EDA_Output/Missing_Values.csv")

print("\nMISSING VALUES")
print(missing[missing > 0])

# 2. Class distribution
class_counts = df["class"].value_counts()
class_counts.to_csv("EDA_Output/Class_Distribution.csv")

print("\nCLASS DISTRIBUTION")
print(class_counts)

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="class")
plt.title("Mushroom Class Distribution")
plt.xlabel("Class (e = edible, p = poisonous)")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig("EDA_Output/Class_Distribution.png")
plt.close()

# 3. Categorical feature distributions
for feature in ["cap-shape", "cap-color", "odor", "gill-size", "habitat"]:
    plt.figure(figsize=(9, 4))
    sns.countplot(data=df, x=feature, order=df[feature].value_counts().index)
    plt.title("Distribution of " + feature)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("EDA_Output/" + feature.replace("-", "_") + "_Distribution.png")
    plt.close()

# 4. Compare feature values across classes
for feature in ["odor", "cap-color", "gill-size", "habitat"]:
    table = pd.crosstab(df[feature], df["class"])
    table.to_csv(
        "EDA_Output/" + feature.replace("-", "_") + "_By_Class.csv"
    )

# 5. Prepare feature data
X = df.drop(columns=["class"]).replace("?", "missing")
y = df["class"]

# One-hot encoding
onehot = OneHotEncoder(handle_unknown="ignore")
X_onehot = onehot.fit_transform(X)

print("\nONE-HOT ENCODING")
print("Original feature count:", X.shape[1])
print("One-hot encoded feature count:", X_onehot.shape[1])

# Ordinal encoding for comparison
ordinal = OrdinalEncoder()
X_ordinal = ordinal.fit_transform(X)

print("\nORDINAL ENCODING")
print("Encoded feature count:", X_ordinal.shape[1])

# 6. Scaling ordinal encoded data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ordinal)

print("\nSCALING")
print("Scaled data shape:", X_scaled.shape)
print("Mean of first scaled feature:", round(X_scaled[:, 0].mean(), 4))

# 7. PCA on ordinal encoded data
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA RESULTS")
print("Original dimensions:", X_scaled.shape[1])
print("Reduced dimensions:", X_pca.shape[1])
print(
    "Explained variance retained:",
    round(pca.explained_variance_ratio_.sum() * 100, 2), "%"
)

# Save PCA output
pca_df = pd.DataFrame(
    X_pca,
    columns=["PC" + str(i + 1) for i in range(X_pca.shape[1])]
)
pca_df["class"] = y.values
pca_df.to_csv("EDA_Output/PCA_Reduced_Data.csv", index=False)

# Save encoding comparison
comparison = pd.DataFrame({
    "Method": ["Original categorical", "Ordinal encoding", "One-hot encoding", "PCA"],
    "Feature_Count": [
        X.shape[1], X_ordinal.shape[1],
        X_onehot.shape[1], X_pca.shape[1]
    ]
})
comparison.to_csv("EDA_Output/Encoding_Comparison.csv", index=False)

# Save summary
with open("EDA_Output/EDA_Summary.txt", "w") as f:
    f.write("MUSHROOM DATASET EDA AND PREPROCESSING\n")
    f.write("Total records: " + str(df.shape[0]) + "\n")
    f.write("Total attributes: " + str(df.shape[1]) + "\n")
    f.write("Missing stalk-root values: " + str(missing["stalk-root"]) + "\n")
    f.write("Edible records: " + str(class_counts.get("e", 0)) + "\n")
    f.write("Poisonous records: " + str(class_counts.get("p", 0)) + "\n")
    f.write("One-hot features: " + str(X_onehot.shape[1]) + "\n")
    f.write("PCA components: " + str(X_pca.shape[1]) + "\n")
    f.write(
        "PCA variance retained: "
        + str(round(pca.explained_variance_ratio_.sum() * 100, 2))
        + "%\n"
    )

print("\nEDA AND PREPROCESSING COMPLETED!")
print("All output files are in the EDA_Output folder.")

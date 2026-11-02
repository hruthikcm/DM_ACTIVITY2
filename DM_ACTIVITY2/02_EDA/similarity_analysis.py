import pandas as pd

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

# Exclude class because this is a feature comparison
features = df.drop(columns=["class"]).replace("?", "missing")

# Compare the first two records
record1 = features.iloc[0]
record2 = features.iloc[1]

matches = (record1 == record2).sum()
total = len(features.columns)

similarity = matches / total
dissimilarity = 1 - similarity

print("CATEGORICAL SIMILARITY ANALYSIS")
print("--------------------------------")
print("Record 1:", 1)
print("Record 2:", 2)
print("Total features compared:", total)
print("Matching features:", matches)
print("Different features:", total - matches)
print("Simple Matching Similarity:", round(similarity, 4))
print("Simple Matching Dissimilarity:", round(dissimilarity, 4))
print("Similarity percentage:", round(similarity * 100, 2), "%")
print("Dissimilarity percentage:", round(dissimilarity * 100, 2), "%")

print("\nFeature comparison:")
for feature in features.columns:
    result = "Match" if record1[feature] == record2[feature] else "Different"
    print(feature, ":", record1[feature], "|", record2[feature], "=>", result)
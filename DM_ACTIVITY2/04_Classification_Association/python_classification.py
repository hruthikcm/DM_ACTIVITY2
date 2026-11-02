
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# 1. Load original dataset
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

# Treat unknown values as a separate category
df = df.replace("?", "missing")

X = df.drop(columns=["class"])
y = df["class"]

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown="ignore")
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])
print("Encoded feature count:", X_train_encoded.shape[1])

results = []

def evaluate_model(name, model, start_time):
    predictions = model.predict(X_test_encoded)
    elapsed = time.time() - start_time

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label="p")
    recall = recall_score(y_test, predictions, pos_label="p")
    f1 = f1_score(y_test, predictions, pos_label="p")

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Execution_Time": elapsed
    })

    print("\n", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("Execution time:", round(elapsed, 4), "seconds")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=["e", "p"]))

# 4. Decision Tree
start = time.time()
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_encoded, y_train)
evaluate_model("Decision Tree", dt, start)

# 5. Rule-based classification: OneR
# Select the single feature with the lowest training error.
best_feature = None
best_rules = None
best_error = float("inf")

for feature in X_train.columns:
    temp = pd.DataFrame({
        "feature": X_train[feature],
        "target": y_train
    })

    rules = temp.groupby("feature")["target"].agg(
        lambda values: values.value_counts().idxmax()
    )

    predictions = X_train[feature].map(rules)
    error = (predictions != y_train).mean()

    if error < best_error:
        best_error = error
        best_feature = feature
        best_rules = rules

class OneRModel:
    def __init__(self, feature, rules, default_class):
        self.feature = feature
        self.rules = rules
        self.default_class = default_class

    def predict(self, encoded_data):
        # OneR uses the original categorical feature values.
        raise NotImplementedError

# Apply learned OneR rules to the original test feature
default_class = y_train.value_counts().idxmax()
oner_predictions = X_test[best_feature].map(best_rules).fillna(default_class)

oner_accuracy = accuracy_score(y_test, oner_predictions)
oner_precision = precision_score(y_test, oner_predictions, pos_label="p")
oner_recall = recall_score(y_test, oner_predictions, pos_label="p")
oner_f1 = f1_score(y_test, oner_predictions, pos_label="p")

results.append({
    "Algorithm": "OneR Rule-Based",
    "Accuracy": oner_accuracy,
    "Precision": oner_precision,
    "Recall": oner_recall,
    "F1_Score": oner_f1,
    "Execution_Time": 0
})

print("\nOneR Rule-Based Classification")
print("Selected feature:", best_feature)
print("Learned rules:")
for value, label in best_rules.items():
    print(value, "->", label)
print("Accuracy:", round(oner_accuracy, 4))
print("Precision:", round(oner_precision, 4))
print("Recall:", round(oner_recall, 4))
print("F1 Score:", round(oner_f1, 4))
print("Confusion matrix:")
print(confusion_matrix(y_test, oner_predictions, labels=["e", "p"]))

# 6. K-Nearest Neighbors
start = time.time()
knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute")
knn.fit(X_train_encoded, y_train)
evaluate_model("KNN (K=5)", knn, start)

# 7. Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv("Python_Classification_Comparison.csv", index=False)

print("\nFinal Algorithm Comparison")
print(results_df.to_string(index=False))

print("\nResults saved to Python_Classification_Comparison.csv")
print("Python classification completed successfully!")

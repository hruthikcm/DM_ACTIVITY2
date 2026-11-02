
import pandas as pd
import time
from mlxtend.frequent_patterns import apriori, association_rules

# Load original categorical Mushroom dataset
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
df = df.replace("?", "missing")

# Convert each categorical value into an item
transactions = pd.DataFrame()

for column in columns:
    for value in df[column].unique():
        item_name = column + "=" + str(value)
        transactions[item_name] = (df[column] == value)

transactions = transactions.astype(bool)
# Convert transactions to Boolean values
transactions = transactions.astype(bool)

print("Dataset records:", len(df))
print("Number of transaction items:", transactions.shape[1])

# Run Apriori
start = time.time()

frequent_itemsets = apriori(
    transactions,
    min_support=0.05,
    use_colnames=True,
    max_len=3
)

elapsed = time.time() - start

# Generate association rules
# Generate association rules safely
if frequent_itemsets.empty:
    rules = pd.DataFrame()
else:
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.80
    )
# Save outputs
frequent_itemsets.to_csv("Python_Apriori_Frequent_Itemsets.csv", index=False)
rules.to_csv("Python_Apriori_Association_Rules.csv", index=False)

print("\nPYTHON APRIORI RESULTS")
print("Frequent itemsets:", len(frequent_itemsets))
print("Association rules:", len(rules))
print("Execution time:", round(elapsed, 3), "seconds")

print("\nTop 10 frequent itemsets:")
print(
    frequent_itemsets.sort_values(
        "support", ascending=False
    ).head(10).to_string(index=False)
)

print("\nTop 10 association rules:")
if not rules.empty:
    print(
        rules.sort_values(
            "confidence", ascending=False
        ).head(10).to_string(index=False)
    )
else:
    print("No association rules met the confidence threshold.")

print("\nPython Apriori completed successfully!")


import pandas as pd
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

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

print("Dataset shape:", df.shape)

df = df.fillna("missing")
df = df.replace("?", "missing")

transactions = []

for _, row in df.iterrows():
    transaction = [
        f"{column}={value}"
        for column, value in row.items()
    ]
    transactions.append(transaction)

encoder = TransactionEncoder()
encoded_array = encoder.fit(transactions).transform(transactions)

encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.columns_
)

start_time = time.time()

frequent_itemsets = fpgrowth(
    encoded_df,
    min_support=0.05,
    use_colnames=True,
    max_len=3
)

execution_time = time.time() - start_time

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.80
)

frequent_itemsets.to_csv(
    "FP_Growth_Frequent_Itemsets.csv", index=False
)

rules.to_csv(
    "FP_Growth_Association_Rules.csv", index=False
)

print("\nFrequent Itemsets:")
print(frequent_itemsets.head(20).to_string(index=False))

print("\nAssociation Rules:")
print(rules.head(20).to_string(index=False))

print("\nTotal frequent itemsets:", len(frequent_itemsets))
print("Total association rules:", len(rules))
print("Execution time:", round(execution_time, 3), "seconds")

print("\nFP-Growth completed successfully!")

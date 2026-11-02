
import pandas as pd
import sqlite3

# Read the original Mushroom dataset
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
df = df.fillna("?").replace("?", "missing")

# Create SQLite database
conn = sqlite3.connect("mushroom_warehouse.db")
cur = conn.cursor()

# Create dimension tables
cur.executescript("""
DROP TABLE IF EXISTS FactCharacteristic;
DROP TABLE IF EXISTS DimFeatureValue;
DROP TABLE IF EXISTS DimFeature;
DROP TABLE IF EXISTS DimClass;
DROP TABLE IF EXISTS DimMushroom;

CREATE TABLE DimClass (
    class_id INTEGER PRIMARY KEY,
    class_label TEXT UNIQUE
);

CREATE TABLE DimMushroom (
    mushroom_id INTEGER PRIMARY KEY,
    class_id INTEGER,
    FOREIGN KEY (class_id) REFERENCES DimClass(class_id)
);

CREATE TABLE DimFeature (
    feature_id INTEGER PRIMARY KEY,
    feature_name TEXT UNIQUE,
    feature_group TEXT
);

CREATE TABLE DimFeatureValue (
    value_id INTEGER PRIMARY KEY,
    feature_id INTEGER,
    value_label TEXT,
    FOREIGN KEY (feature_id) REFERENCES DimFeature(feature_id)
);

CREATE TABLE FactCharacteristic (
    mushroom_id INTEGER,
    feature_id INTEGER,
    value_id INTEGER,
    FOREIGN KEY (mushroom_id) REFERENCES DimMushroom(mushroom_id),
    FOREIGN KEY (feature_id) REFERENCES DimFeature(feature_id),
    FOREIGN KEY (value_id) REFERENCES DimFeatureValue(value_id)
);
""")

# Insert class dimension
classes = sorted(df["class"].unique())
class_ids = {}

for i, label in enumerate(classes, start=1):
    cur.execute(
        "INSERT INTO DimClass VALUES (?, ?)", (i, label)
    )
    class_ids[label] = i

# Insert mushroom records
for mushroom_id, label in enumerate(df["class"], start=1):
    cur.execute(
        "INSERT INTO DimMushroom VALUES (?, ?)",
        (mushroom_id, class_ids[label])
    )

# Feature groups for warehouse analysis
feature_groups = {
    "cap": ["cap-shape", "cap-surface", "cap-color"],
    "gill": ["gill-attachment", "gill-spacing", "gill-size", "gill-color"],
    "stalk": [
        "stalk-shape", "stalk-root", "stalk-surface-above-ring",
        "stalk-surface-below-ring", "stalk-color-above-ring",
        "stalk-color-below-ring"
    ],
    "other": [
        "bruises", "odor", "veil-type", "veil-color",
        "ring-number", "ring-type", "spore-print-color",
        "population", "habitat"
    ]
}

# Insert features and characteristic values
feature_id = 0
value_id = 0

for feature in columns[1:]:
    feature_id += 1

    group = next(
        (g for g, names in feature_groups.items() if feature in names),
        "other"
    )

    cur.execute(
        "INSERT INTO DimFeature VALUES (?, ?, ?)",
        (feature_id, feature, group)
    )

    values = sorted(df[feature].unique())
    value_lookup = {}

    for value in values:
        value_id += 1
        cur.execute(
            "INSERT INTO DimFeatureValue VALUES (?, ?, ?)",
            (value_id, feature_id, value)
        )
        value_lookup[value] = value_id

    for mushroom_id, value in enumerate(df[feature], start=1):
        cur.execute(
            "INSERT INTO FactCharacteristic VALUES (?, ?, ?)",
            (mushroom_id, feature_id, value_lookup[value])
        )

conn.commit()

print("Mushroom data warehouse created successfully!")
print("Total mushroom records:", len(df))
print("Total features:", len(columns) - 1)

# OLAP Query 1: Class distribution
print("\nOLAP 1 - Class distribution")
for row in cur.execute("""
    SELECT c.class_label, COUNT(*) AS total
    FROM DimMushroom m
    JOIN DimClass c ON m.class_id = c.class_id
    GROUP BY c.class_label
"""):
    print(row)

# OLAP Query 2: Characteristic values by class
print("\nOLAP 2 - Characteristic values by class")
for row in cur.execute("""
    SELECT f.feature_name, v.value_label,
           c.class_label, COUNT(*) AS total
    FROM FactCharacteristic fc
    JOIN DimMushroom m ON fc.mushroom_id = m.mushroom_id
    JOIN DimClass c ON m.class_id = c.class_id
    JOIN DimFeature f ON fc.feature_id = f.feature_id
    JOIN DimFeatureValue v ON fc.value_id = v.value_id
    GROUP BY f.feature_name, v.value_label, c.class_label
    ORDER BY f.feature_name, total DESC
    LIMIT 20
"""):
    print(row)

# OLAP Query 3: Feature group analysis
print("\nOLAP 3 - Feature group totals")
for row in cur.execute("""
    SELECT f.feature_group, c.class_label, COUNT(*) AS total
    FROM FactCharacteristic fc
    JOIN DimMushroom m ON fc.mushroom_id = m.mushroom_id
    JOIN DimClass c ON m.class_id = c.class_id
    JOIN DimFeature f ON fc.feature_id = f.feature_id
    GROUP BY f.feature_group, c.class_label
    ORDER BY f.feature_group
"""):
    print(row)

conn.close()
print("\nWarehouse and OLAP analysis completed!")

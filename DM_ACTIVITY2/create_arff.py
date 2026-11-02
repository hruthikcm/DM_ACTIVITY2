
from pathlib import Path

# Original UCI dataset location
folder = Path(__file__).parent
input_file = folder / "01_Dataset" / "mushroom" / "agaricus-lepiota.data"
output_file = folder / "mushroom.arff"

# Mushroom dataset attributes and their possible values
attributes = {
    "class": "p,e",
    "cap-shape": "b,c,x,f,k,s",
    "cap-surface": "f,g,y,s",
    "cap-color": "n,b,c,g,r,p,u,e,w,y",
    "bruises": "t,f",
    "odor": "a,l,c,y,f,m,n,p,s",
    "gill-attachment": "a,d,f,n",
    "gill-spacing": "c,w,d",
    "gill-size": "b,n",
    "gill-color": "k,n,b,h,g,r,o,p,u,e,w,y",
    "stalk-shape": "e,t",
    "stalk-root": "b,c,u,e,z,r",
    "stalk-surface-above-ring": "f,y,k,s",
    "stalk-surface-below-ring": "f,y,k,s",
    "stalk-color-above-ring": "n,b,c,g,o,p,e,w,y",
    "stalk-color-below-ring": "n,b,c,g,o,p,e,w,y",
    "veil-type": "p,u",
    "veil-color": "n,o,w,y",
    "ring-number": "n,o,t",
    "ring-type": "c,e,f,l,n,p,s,z",
    "spore-print-color": "k,n,b,h,r,o,u,w,y",
    "population": "a,c,n,s,v,y",
    "habitat": "g,l,m,p,u,w,d"
}

if not input_file.exists():
    raise FileNotFoundError(f"Dataset not found: {input_file}")

with input_file.open("r", encoding="utf-8") as source:
    rows = [line.strip() for line in source if line.strip()]

with output_file.open("w", encoding="utf-8") as output:
    output.write("@relation mushroom\n\n")

    for name, values in attributes.items():
        output.write(f"@attribute {name} {{{values}}}\n")

    output.write("\n@data\n")

    for line_number, row in enumerate(rows, start=1):
        values = row.split(",")

        if len(values) != 23:
            raise ValueError(
                f"Row {line_number} has {len(values)} values, not 23."
            )

        # Convert missing stalk-root values from ? to WEKA missing values
        values = ["?" if value == "?" else value for value in values]

        output.write(",".join(values) + "\n")

print("Conversion completed successfully!")
print("Total records:", len(rows))
print("Total attributes:", len(attributes))
print("Saved file:", output_file)

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw_data.csv", sep=",")

bins_counts = df["AVG BPM"].value_counts(bins=20).sort_index()

# formatta le etichette dei bin
labels = [
    f"({round(interval.left, 1)}, {round(interval.right, 1)}]"
    for interval in bins_counts.index
]

plt.figure(figsize=(10, 5))
plt.bar(labels, bins_counts.values)
plt.xlabel("Bins (BPM)")
plt.ylabel("Count")
plt.title("Distribuzione per bins")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

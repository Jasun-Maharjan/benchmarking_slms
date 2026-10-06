import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("llama3.2_benchmark_results.csv")

print(df.shape)
print(df.isna().sum())
print(df.groupby("prompt")[["promptTokens","outputTokens"]].median())

summary =  (df.groupby("prompt")[["ttftSeconds", "tps_wall", "tps_ollama", "totalSeconds"]]
             .agg(["median", "std"]).round(3))
print(summary)

order = ["short","medium","large"]
fig, axes = plt.subplots(1,3)

for ax, col, title in zip(axes,
                          ["ttftSeconds", "tps_wall", "totalSeconds"],
                          ["Time to first token (s)", "Tokens/sec (decode)", "Total latency (s)"]):
    data = [df[df["prompt"]==p][col].dropna() for p in order]
    ax.boxplot(data, tick_labels = order)
    ax.set_title(title)

plt.suptitle("llama3.2:3b baseline (Q4, local)")
plt.tight_layout()
plt.show()
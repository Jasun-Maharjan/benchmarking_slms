import pandas as pd

df = pd.read_csv("llama3.2_benchmark_results.csv")

print(df.shape)
print(df.isna().sum())
print(df.groupby("prompt")[["promptTokens","outputTokens"]].median())

summary =  (df.groupby("prompt")[["ttftSeconds", "tps_wall", "tps_ollama", "totalSeconds"]]
             .agg(["median", "std"]).round(3))
print(summary)
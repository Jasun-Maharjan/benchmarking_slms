from benchmark import benchmark
from pathlib import Path
import pandas as pd

Prompts = {
    "short":"What does photosyntesis mean? Expain in one sentence",
    "medium":"Explain how CPU cache work, including L1, L2 and L3, in about 150 words.",
    "large": "The quick brown fox jumps over the lazy lion. " * 100
             + "\nSummarize the text above in two sentences."
}

Runs = 10

benchmark("Hello.")
rows = []

for name, prompt in Prompts.items():
    for i in range(Runs):
        result = benchmark(prompt)
        result.update({"prompt":name, "run": i})
        rows.append(result)
        print(f"{name} run {i+1}/{Runs} completed.")

df = pd.DataFrame(rows)
out = Path(__file__).parent / "llama3.2_benchmark_results.csv"
df.to_csv(out, index=False)
summary = df.groupby("prompt")[["ttftSeconds","tps_wall","totalSeconds"]].agg(["median", "std"])
print(summary.round(3))
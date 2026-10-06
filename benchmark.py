import ollama
import time

MODEL = "llama3.2:3b"

def ns_to_s(value):
    return value / 1e9 if value is not None else None

def benchmark(prompt, model = MODEL):
    start = time.perf_counter()
    first = None
    final = None

    for chunk in ollama.chat(
        model = model,
        stream = True,
        messages = [{
            "role": "user",
            "content": prompt
        }]
    ):
        if first is None and chunk.message.content:
            first = time.perf_counter()
        if chunk.done:
            final = chunk

    end = time.perf_counter()

    return {
        "ttftSeconds": first-start,
        "totalSeconds": end-start,
        "outputTokens": final.eval_count,
        "promptTokens": final.prompt_eval_count,
        "tps_wall": (final.eval_count - 1) / (end - first) if final.eval_count else None,
        "tps_ollama": (final.eval_count / (final.eval_duration / 1e9)
                       if final.eval_count and final.eval_duration else None),
        "load_s": ns_to_s(final.load_duration),
    }

if __name__ == "__main__":
    result = benchmark("Explain what a AI model is in 100 words.")
    for k, v in result.items():
        print(f"{k}:{v:.3f}" 
              if isinstance(v, float)
              else f"{k}:{v}")
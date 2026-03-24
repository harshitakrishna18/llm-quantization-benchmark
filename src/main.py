import argparse
import json
import os

from src.load_model import load_baseline, load_8bit, load_4bit
from src.benchmark import benchmark_generation
from src.utils import get_model_size_mb

PROMPTS = [
    "The future of machine learning is",
    "Quantization helps large language models by",
    "In an IoT anomaly detection pipeline,"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "8bit", "4bit"], required=True)
    parser.add_argument("--model_name", default="distilgpt2")
    args = parser.parse_args()

    if args.mode == "baseline":
        tokenizer, model, load_time = load_baseline(args.model_name)
    elif args.mode == "8bit":
        tokenizer, model, load_time = load_8bit(args.model_name)
    else:
        tokenizer, model, load_time = load_4bit(args.model_name)

    size_mb = get_model_size_mb(model)

    generations = []
    for prompt in PROMPTS:
        result = benchmark_generation(model, tokenizer, prompt)
        generations.append(result)

    avg_latency = sum(x["latency_in_sec"] for x in generations) / len(generations)
    avg_tps = sum(x["tokens_per_sec"] for x in generations) / len(generations)

    summary = {
        "mode": args.mode,
        "model_name": args.model_name,
        "load_time_s": load_time,
        "model_size_mb": size_mb,
        "avg_latency_s": avg_latency,
        "avg_tokens_per_sec": avg_tps,
        "samples": generations,
    }

    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{args.mode}_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
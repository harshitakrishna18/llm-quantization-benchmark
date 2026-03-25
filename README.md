# LLM Quantization Benchmark

A small project to compare baseline vs quantized small causal language models and study inference tradeoffs.

## Goal
Evaluate how quantization affects:

- model load time
- inference latency
- approximate memory footprint
- qualitative output quality

## Why this project?
This project is meant to build practical intuition around LLM inference optimization, especially in the context of:

- memory-constrained serving
- latency / throughput tradeoffs
- prefill vs decode behavior
- quantization as an inference optimization lever

## Initial setup
Starting with a small causal LM (`distilgpt2`) for fast iteration and simple benchmarking.

## Planned comparisons
- Baseline model
- Quantized model (CPU dynamic quantization or 8-bit / 4-bit path later)

## Metrics
- Load time
- Per-prompt generation latency
- Number of output tokens
- Qualitative output comparison

## Prompts
1. The future of machine learning is
2. Quantization helps large language models by
3. In an IoT anomaly detection pipeline,

## Next steps
- [x] Project skeleton
- [x] Baseline benchmark
- [x] Quantized benchmark
- [ ] Compare results and summarize tradeoffs

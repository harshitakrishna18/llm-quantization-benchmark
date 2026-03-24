## Baseline Results (distilgpt2)

### Aggregate Metrics

| Metric | Value |
|---|---:|
| Load Time (s) | 1.518 |
| Model Size (MB) | 312.472 |
| Average Latency (s) | 1.031 |
| Average Tokens/sec | 56.923 |

### Per-Sample Metrics

| Sample | Prompt | Latency (s) | New Tokens | Tokens/sec |
|---|---|---:|---:|---:|
| 1 | The future of machine learning is | 1.674 | 50 | 29.866 |
| 2 | Quantization helps large language models by | 0.693 | 50 | 72.134 |
| 3 | In an IoT anomaly detection pipeline, | 0.727 | 50 | 68.768 |



## 8-bit Results (distilgpt2)

### Aggregate Metrics

| Metric | Value |
|---|---:|
| Load Time (s) | 4.207 |
| Model Size (MB) | 190.972 |
| Average Latency (s) | 4.877 |
| Average Tokens/sec | 14.704 |

### Per-Sample Metrics

| Sample | Prompt | Latency (s) | New Tokens | Tokens/sec |
|---|---|---:|---:|---:|
| 1 | The future of machine learning is | 9.012 | 50 | 5.548 |
| 2 | Quantization helps large language models by | 3.588 | 50 | 13.934 |
| 3 | In an IoT anomaly detection pipeline, | 2.030 | 50 | 24.629 |

## 4-bit Results (distilgpt2)

### Aggregate Metrics

| Metric | Value |
|---|---:|
| Load Time (s) | 7.033 |
| Model Size (MB) | 170.722 |
| Average Latency (s) | 7.910 |
| Average Tokens/sec | 6.363 |

### Per-Sample Metrics

| Sample | Prompt | Latency (s) | New Tokens | Tokens/sec |
|---|---|---:|---:|---:|
| 1 | The future of machine learning is | 8.735 | 50 | 5.724 |
| 2 | Quantization helps large language models by | 7.842 | 50 | 6.376 |
| 3 | In an IoT anomaly detection pipeline, | 7.154 | 50 | 6.989 |
| Model | Strategy | Unsupported Code Rate | Parse Failure Rate | Overflow Rate | Truncation Rate | Mean Codes Emitted | n_p5 | n_p8 | Mean Latency (s) | GPU | Candidate Recall@k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mock | Zero-Shot | 90.0% | 0.0% | 0.0% | 0.0% | 3.00 | 0 | 0 | 0.02 | cpu | n/a |
| mock | Few-Shot (k=5) | 90.0% | 0.0% | 0.0% | 0.0% | 3.00 | 0 | 0 | 0.02 | cpu | n/a |
| mock | RAG | 90.0% | 0.0% | 0.0% | 0.0% | 3.00 | 0 | 0 | 0.02 | cpu | 42.8% |
| mock | KG | 90.0% | 0.0% | 0.0% | 0.0% | 3.00 | 0 | 0 | 0.02 | cpu | 23.2% |
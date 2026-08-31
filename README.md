# When Spectral Attention Graphs Fail

Code for **"When Spectral Attention Graphs Fail: A Confound-Aware Study of Reasoning in LLM Attention Graphs."**

## Overview

This repository contains the experimental pipeline for analyzing spectral features of layer-wise attention graphs as signals of reasoning correctness. Experiments are conducted on GSM8K and ARC-Challenge using Qwen2.5-0.5B-Instruct.

## Results

Across both benchmarks, spectral attention-graph differences between correct and incorrect reasoning are either substantially explained by response length, attention-sink mass, and effective rank, or fail to survive multiple-comparison correction. Dynamic (layer-to-layer) features do not outperform static features or a confound-only baseline.

## Notebooks

### GSM8K
1. `01_gsm8k_generation.ipynb` — response generation and tensor extraction
2. `02_gsm8k_analysis.ipynb` — correctness labeling and audit
3. `03_gsm8k_features.ipynb` — spectral features, confound analysis, and classification

### ARC-Challenge
4. `04_arc_generation.ipynb` — response generation and tensor extraction
5. `05_arc_analysis.ipynb` — correctness labeling and audit
6. `06_arc_features.ipynb` — spectral features, confound analysis, and classification

### Final analyses
7. `07_final_statistics.ipynb` — multiple-testing correction and robustness analyses
8. `08_cleanup.ipynb` — final verification analyses

## Requirements

Notebooks were run on Kaggle (2× NVIDIA T4 GPU for generation steps). Key dependencies: `transformers`, `datasets`, `torch`, `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`.

## License

[MIT](LICENSE)
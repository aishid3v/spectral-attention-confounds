# When Spectral Attention Graphs Fail

Code for **"When Spectral Attention Graphs Fail: A Confound-Aware Study of Reasoning in LLM Attention Graphs."**

## Overview

This repository contains the experimental pipeline for analyzing spectral features of layer-wise attention graphs as signals of reasoning correctness. Experiments are conducted on GSM8K and ARC-Challenge using Qwen2.5-0.5B-Instruct.

## Notebooks

### GSM8K
1. `01-gsm8k-generation.ipynb` — response generation and tensor extraction
2. `02-gsm8k-analysis.ipynb` — correctness labeling and audit
3. `03-gsm8k-features.ipynb` — spectral features, confound analysis, and classification

### ARC-Challenge
4. `04-arc-generation.ipynb` — response generation and tensor extraction
5. `05-arc-analysis.ipynb` — correctness labeling and audit
6. `06-arc-features.ipynb` — spectral features, confound analysis, and classification

### Final analyses
7. `07-final-statistics.ipynb` — multiple-testing correction and robustness analyses
8. `08-cleanup.ipynb` — final verification analyses

## Requirements

Notebooks were run on Kaggle (2× NVIDIA T4 GPU for generation steps). Key dependencies: `transformers`, `datasets`, `torch`, `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`.

## Results

Across both benchmarks, spectral attention-graph differences between correct and incorrect reasoning are either substantially explained by response length, attention-sink mass, and effective rank, or fail to survive multiple-comparison correction. Dynamic (layer-to-layer) features do not outperform static features or a confound-only baseline.

## Reproducibility

### Environment

Install the required Python dependencies with:

```bash
pip install -r requirements.txt
```

Experiments were run in Kaggle notebooks. Generation and attention/hidden-state extraction used 2× NVIDIA T4 GPUs; subsequent analysis was performed on CPU.

### Notebook run order

Notebooks should be run in the order below. Downstream notebooks require outputs from earlier stages, attached in Kaggle via **Add Data → Notebook Output Files**.

| Order | Notebook | Depends on |
| --- | --- | --- |
| 1 | `01-gsm8k-generation.ipynb` | — |
| 2 | `02-gsm8k-analysis.ipynb` | Output of 1 |
| 3 | `03-gsm8k-features.ipynb` | Outputs of 1 and 2 |
| 4 | `04-arc-generation.ipynb` | — |
| 5 | `05-arc-analysis.ipynb` | Output of 4 |
| 6 | `06-arc-features.ipynb` | Outputs of 4 and 5 |
| 7 | `07-final-statistics.ipynb` | Outputs of 3 and 6 |
| 8 | `08-cleanup.ipynb` | Outputs of 3, 6, and 7 |

### Outputs

Outputs are organized under `/kaggle/working/spectral_v2/` into `generations/`, `features/`, and `results/`.

- **Generation (1, 4):** generates model responses and extracts attention weights and hidden states for each example.
- **Analysis (2, 5):** audits automatically assigned correctness labels, applies manual corrections where necessary, and records the labeling review.
- **Features (3, 6):** computes per-layer spectral diagnostics for the combinatorial and normalized Laplacians, performs confound screening, and evaluates spectral and confound-only classifiers.
- **Final statistics (7):** applies Benjamini-Hochberg correction to the layer-metric tests and performs the GSM8K layer-16 follow-up analysis.
- **Cleanup (8):** performs the final GSM8K layer-16 direct classifier test and ARC-Challenge permutation test.

### Reproducing paper results

- **Table 1 — Confound screening:** produced from the GSM8K and ARC-Challenge feature/confound analyses in notebooks 3 and 6.
- **Table 2 — BH-corrected tests:** produced from the multiple-comparison analysis in notebook 7.
- **Table 3 — Classifier comparison:** produced from the classifier analyses in notebooks 3 and 6, with the ARC-Challenge permutation follow-up in notebook 8.
- **GSM8K layer-16 follow-up:** the final direct classifier test is performed in notebook 8.

### Experimental setup

Experiments use **Qwen2.5-0.5B-Instruct** on 150 fixed-seed examples from each benchmark. Generation uses temperature `T = 0.3`, a maximum response length of 768 tokens, and seed `42`.

After manual label auditing:

| Dataset | Total | Correct | Incorrect |
| --- | ---: | ---: | ---: |
| GSM8K | 150 | 65 | 85 |
| ARC-Challenge | 150 | 58 | 92 |

Classifier evaluation uses 5-fold stratified cross-validation with inner 3-fold cross-validation for regularization selection. Statistical testing uses Benjamini-Hochberg correction at α = 0.05 across the 96 layer-metric tests for each benchmark and Laplacian variant.

### Runtime and hardware

Generation and attention/hidden-state extraction were performed on Kaggle using **2× NVIDIA T4 GPUs**, taking approximately **45–75 minutes per dataset**. Feature computation, confound screening, classification, and significance testing were performed on CPU.

### Known limitations

- We evaluate a single model, Qwen2.5-0.5B-Instruct; spectral signatures may be architecture-dependent.
- Both benchmarks consist of short-form reasoning tasks, so the findings may not generalize to longer-form reasoning.
- The sample size of 150 examples per dataset limits statistical power.
- Confound residualization and classifiers are linear and may not capture nonlinear relationships between confounds and spectral features.
- Symmetrizing the attention graphs discards directional information between tokens.
- Candidate layers used to construct the sink-mass and effective-rank confounds were selected using correctness labels on the full dataset before cross-validation, introducing potential selection leakage. See the paper's Limitations section for this caveat.


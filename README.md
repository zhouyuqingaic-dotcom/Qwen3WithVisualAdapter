# RoMA-Net

**RoMA-Net: Router-guided Multi-scale Visual Residual Adaptation for Medical Visual Question Answering**

RoMA-Net is a router-guided multi-scale visual residual adaptation framework for Medical Visual Question Answering (Med-VQA). It is built on **Qwen3-VL-8B-Instruct** and studies whether frozen visual tokens still contain exploitable residual medical evidence after strong two-stage LoRA adaptation.

The framework keeps both the Qwen3-VL visual backbone and BioMedCLIP frozen. BioMedCLIP is used only as a low-bandwidth cross-modal router to generate soft routing weights for multi-scale visual residual experts. It does **not** inject BioMedCLIP visual tokens, text tokens, retrieval results, external medical knowledge, or semantic memory into the language model.

Repository branch used for the manuscript:

```text
feat/multi-scale-visual-adapter-router
```

## Framework Architecture

<p align="center">
  <img src="./framework_V2.png" alt="RoMA-Net Framework Architecture" width="100%">
</p>

<p align="center">
  <em>Overall architecture of RoMA-Net.</em>
</p>

Repository URL:

```text
https://github.com/zhouyuqingaic-dotcom/RoMA-Net/tree/feat/multi-scale-visual-adapter-router
```

---

## 1. Overview

This repository contains the implementation for:

- Two-stage LoRA medical adaptation
- Stage-1 medical vision-language pre-alignment on MIMIC-CXR
- Stage-2 downstream Med-VQA adaptation
- BioMedCLIP-guided low-bandwidth routing
- Multi-scale visual residual adaptation
- Residual-strength sensitivity analysis
- Routing behavior analysis
- Vanilla Qwen3-VL baseline evaluation
- RoMA-Net checkpoint evaluation
- GPT-assisted semantic judging for open-ended answers

The main datasets used in the manuscript are:

- MIMIC-CXR
- VQA-RAD
- SLAKE
- VQA-Med-2019

---

## 2. Method Summary

Given a medical image and a question, Qwen3-VL first extracts the original visual token sequence. RoMA-Net constructs multi-scale visual residuals from these frozen visual tokens and uses BioMedCLIP only to produce low-dimensional routing weights. The fused visual residual is injected into the original visual tokens as:

```math
\hat{X} = X + \alpha \Delta X
```

where `X` denotes the original Qwen3-VL visual tokens, `Delta X` denotes the fused multi-scale visual residual, and `alpha` controls the residual strength.

When `alpha = 0`, the residual path is disabled and RoMA-Net degenerates to the two-stage LoRA-only baseline.

---

## 3. Repository Structure

```text
RoMA-Net/
├── config/
│   ├── LLM_config.py
│   ├── stage1_train_config_mimic_cxr.py
│   ├── vqa_rad/
│   │   ├── stage2_train_config_vqa_rad.py
│   │   └── stage2_eval_config_vqa_rad.py
│   ├── slake/
│   │   ├── stage2_train_config_slake.py
│   │   └── stage2_eval_config_slake.py
│   └── vqa_med_2019/
│       ├── stage2_train_config_vqa_med_2019.py
│       └── stage2_eval_config_vqa_med_2019.py
├── datas/
│   ├── mimic_cxr_datasets.py
│   ├── vqa_rad_datasets.py
│   ├── slake_datasets.py
│   └── vqa_med_2019_datasets.py
├── training/
│   ├── stage1_trainer_mimic_cxr.py
│   ├── stage2_trainer_vqa_rad.py
│   ├── stage2_trainer_slake.py
│   └── stage2_trainer_vqa_med_2019.py
├── evaling/
│   ├── stage2_eval_qwen_only_vqa_rad.py
│   ├── stage2_eval_qwen_only_slake.py
│   ├── stage2_eval_qwen_only_vqa_med.py
│   ├── stage2_eval_checkpointings_vqa_rad.py
│   ├── stage2_eval_checkpoints_slake.py
│   ├── stage2_eval_checkpoints_vqa_med.py
│   ├── stage2_eval_analys_slake.py
│   ├── stage2_test_checkpoints_slake.py
│   └── stage2_test_checkpoints_vqa_med.py
├── testing/
│   ├── generate_test_official_jsonl.py
│   ├── testing_vqa_rad_official_testing.py
│   ├── testing_slake_datasets.py
│   ├── testing_vqa_med_2019_datasets.py
│   └── testing_vqa_med_2019_dataset_collator.py
├── utils/
│   ├── qwen3vl/
│   ├── biomedclip/
│   ├── data_tools/
│   └── ddp/
├── LLM_api/
│   ├── gpt_5_mini.py
│   └── prompts/
├── requirements.txt
└── README.md
```

---

## 4. Environment Setup

We recommend using a Conda environment. The reported experiments were conducted with Python 3.11.

```bash
conda create -n qwen3vl python=3.11
conda activate qwen3vl
pip install -r requirements.txt
```

The main software environment used in the reported experiments is:

| Package | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.10.0+cu130 |
| TorchVision | 0.25.0+cu130 |
| CUDA | 13.0 |
| Transformers | 5.3.0.dev0 |
| PEFT | 0.18.1 |
| Accelerate | 1.13.0 |
| BitsAndBytes | 0.49.2 |
| Datasets | 4.7.0 |
| Tokenizers | 0.22.2 |
| SentencePiece | 0.2.1 |
| qwen-vl-utils | 0.0.14 |
| NumPy | 2.3.5 |
| Pandas | 3.0.1 |
| Pillow | 12.0.0 |
| OpenAI | 2.29.0 |
| tqdm | 4.67.3 |

The reported experiments were run on a multi-GPU server with:

```text
8 x NVIDIA A40 GPUs
CUDA version: 13.0
PyTorch CUDA build: cu130
```

DeepSpeed and TRL are not required for the reported experiments.

---

## 5. Installation Notes

The exact PyTorch and Transformers versions used in the experiments may require a CUDA-specific or development package index. If the exact versions in `requirements.txt` are unavailable, install compatible versions manually.

Example:

```bash
conda create -n qwen3vl python=3.11
conda activate qwen3vl

# Install PyTorch/TorchVision compatible with your CUDA environment.
# Then install the remaining dependencies.
pip install -r requirements.txt
```

If `flash-attn` is not available on your system, change the attention implementation in the Qwen3-VL loader or configuration from:

```text
flash_attention_2
```

to:

```text
sdpa
```

or:

```text
eager
```

---

## 6. Data Preparation

Dataset files are not redistributed in this repository due to dataset license restrictions. Please download the datasets from their official sources:

- MIMIC-CXR
- VQA-RAD
- SLAKE
- VQA-Med-2019

After downloading the datasets, edit the path fields in the corresponding configuration files.

Main configuration files:

```text
config/stage1_train_config_mimic_cxr.py
config/vqa_rad/stage2_train_config_vqa_rad.py
config/vqa_rad/stage2_eval_config_vqa_rad.py
config/slake/stage2_train_config_slake.py
config/slake/stage2_eval_config_slake.py
config/vqa_med_2019/stage2_train_config_vqa_med_2019.py
config/vqa_med_2019/stage2_eval_config_vqa_med_2019.py
```

The role of each dataset is:

| Dataset | Usage |
|---|---|
| MIMIC-CXR | Stage-1 medical vision-language pre-alignment |
| VQA-RAD | Stage-2 Med-VQA adaptation and evaluation |
| SLAKE | Stage-2 Med-VQA adaptation and evaluation |
| VQA-Med-2019 | Stage-2 Med-VQA adaptation and evaluation |

For SLAKE and VQA-Med-2019, the official training and validation sets are merged for Stage-2 adaptation, and the official test set is used for final reporting.

---

## 7. Model Preparation

The code expects local or accessible paths for:

- Qwen3-VL-8B-Instruct
- BioMedCLIP
- MIMIC-CXR
- VQA-RAD
- SLAKE
- VQA-Med-2019
- Stage-1 LoRA weights for Stage-2 initialization

Please edit the corresponding fields in the configuration files before running training or evaluation.

BioMedCLIP should be organized so that the loader can access files such as:

```text
open_clip_config.json
open_clip_pytorch_model.bin
tokenizer files
```

---

## 8. API Key and Security

GPT-assisted semantic judging is used only for offline evaluation of open-ended answers. It is not used during model training or model inference.

Please provide the API key through an environment variable:

```bash
export GPT_5_MINI_API_KEY=your_api_key
```

Do not hard-code API keys in source files, configuration files, or commit history.

A safe configuration pattern is:

```python
import os

gpt_5_mini_key = os.environ.get("GPT_5_MINI_API_KEY", "")
```

If an API key has ever been committed to the repository, rotate or revoke that key and remove the secret from the public history before releasing the code.

---

## 9. Training

The training scripts import Python configuration files directly. Before running a script, edit the corresponding config file to set dataset paths, model paths, output directories, seeds, and residual-routing settings.

### 9.1 Stage 1: MIMIC-CXR Medical Vision-Language Pre-alignment

Single-process run:

```bash
python training/stage1_trainer_mimic_cxr.py
```

Multi-GPU run with 8 GPUs:

```bash
torchrun --nproc_per_node=8 training/stage1_trainer_mimic_cxr.py
```

Main config:

```text
config/stage1_train_config_mimic_cxr.py
```

Stage 1 saves LoRA weights and the visual adapter state under the configured output directory.

---

### 9.2 Stage 2: VQA-RAD Adaptation

Single-process run:

```bash
python training/stage2_trainer_vqa_rad.py
```

Multi-GPU run with 8 GPUs:

```bash
torchrun --nproc_per_node=8 training/stage2_trainer_vqa_rad.py
```

Main config:

```text
config/vqa_rad/stage2_train_config_vqa_rad.py
```

---

### 9.3 Stage 2: SLAKE Adaptation

Single-process run:

```bash
python training/stage2_trainer_slake.py
```

Multi-GPU run with 8 GPUs:

```bash
torchrun --nproc_per_node=8 training/stage2_trainer_slake.py
```

Main config:

```text
config/slake/stage2_train_config_slake.py
```

---

### 9.4 Stage 2: VQA-Med-2019 Adaptation

Single-process run:

```bash
python training/stage2_trainer_vqa_med_2019.py
```

Multi-GPU run with 8 GPUs:

```bash
torchrun --nproc_per_node=8 training/stage2_trainer_vqa_med_2019.py
```

Main config:

```text
config/vqa_med_2019/stage2_train_config_vqa_med_2019.py
```

---

## 10. Evaluation

Evaluation scripts also import Python configuration files directly. Please edit the corresponding evaluation config file before running evaluation.

### 10.1 Vanilla Qwen3-VL Baselines

VQA-RAD:

```bash
python evaling/stage2_eval_qwen_only_vqa_rad.py
```

SLAKE:

```bash
python evaling/stage2_eval_qwen_only_slake.py
```

VQA-Med-2019:

```bash
python evaling/stage2_eval_qwen_only_vqa_med.py
```

---

### 10.2 RoMA-Net Checkpoint Evaluation

VQA-RAD:

```bash
python evaling/stage2_eval_checkpointings_vqa_rad.py
```

SLAKE:

```bash
python evaling/stage2_eval_checkpoints_slake.py
```

VQA-Med-2019:

```bash
python evaling/stage2_eval_checkpoints_vqa_med.py
```

---

### 10.3 Test Checkpoint Evaluation

SLAKE:

```bash
python evaling/stage2_test_checkpoints_slake.py
```

VQA-Med-2019:

```bash
python evaling/stage2_test_checkpoints_vqa_med.py
```

---

### 10.4 Routing Behavior Analysis

SLAKE routing analysis:

```bash
python evaling/stage2_eval_analys_slake.py
```

This script can be used to inspect the routing weights assigned to the Global, Local, and Region residual experts.

---

## 11. Evaluation Protocol

The evaluation includes:

- Closed-ended exact matching
- Open-ended normalized exact matching
- GPT-assisted semantic judging for open-ended exact-match failures
- Overall accuracy calculation
- Residual-strength sensitivity analysis
- Routing behavior analysis

Closed-ended questions are evaluated by normalized exact matching. Open-ended questions are first evaluated by normalized exact matching. Open-ended exact-match failures are then checked by GPT-assisted semantic judging.

The semantic judge is used only during offline evaluation and is not used during training or inference.

---

## 12. Answer Normalization

Before exact matching, answers are normalized by rules such as:

- Lowercasing
- Removing redundant punctuation
- Removing extra spaces
- Normalizing yes/no answers
- Removing wrapper phrases when appropriate
- Applying dataset-specific answer cleaning rules

Dataset-specific normalization utilities are located under:

```text
utils/data_tools/prompt_cleaning/
```

---

## 13. Residual Strength Analysis

The residual-strength coefficient `alpha` controls the magnitude of visual residual injection:

```math
\hat{X} = X + \alpha \Delta X
```

The manuscript evaluates residual strength values:

```text
alpha = 0.0, 0.1, 0.2, ..., 1.0
```

When `alpha = 0`, the visual residual path is disabled and the model corresponds to the two-stage LoRA-only baseline.

---

## 14. Reproducibility Notes

The reported experiments use:

- Qwen3-VL-8B-Instruct as the base MLLM
- LoRA for parameter-efficient adaptation
- Frozen Qwen3-VL visual backbone
- Frozen BioMedCLIP encoders
- Multi-scale visual residual experts
- BioMedCLIP-guided low-bandwidth routing
- Residual-strength coefficient `alpha`
- GPT-assisted semantic judging only for offline open-ended evaluation

Important implementation notes:

- BioMedCLIP is used only to generate routing weights.
- BioMedCLIP does not provide answer content.
- No external retrieval module is used.
- No BioMedCLIP visual token or text token is injected into the LLM decoder.
- Dataset files are not redistributed.
- API keys must be provided only through environment variables.

---

## 15. Main Results

Under the same-backbone controlled setting, RoMA-Net provides limited but observable visual-side improvements beyond a strong two-stage LoRA-only baseline.

| Dataset | LoRA-only Overall Acc. | RoMA-Net Overall Acc. | Gain |
|---|---:|---:|---:|
| VQA-RAD | 63.19 | 64.30 | +1.11 |
| SLAKE | 88.16 | 88.54 | +0.38 |
| VQA-Med-2019 | 61.20 | 61.40 | +0.20 |

The largest improvement appears on VQA-RAD open-ended questions, where Open Accuracy increases from 42.46 to 46.93.

---

## 16. Citation

If you use this repository, please cite our work:

```bibtex
@article{zhou2026romanet,
  title={RoMA-Net: Router-guided Multi-scale Visual Residual Adaptation for Medical Visual Question Answering},
  author={Zhou, Yuqing and Yan, Feng and Xu, Pengfei and Sun, Qihui},
  journal={},
  year={2026}
}
```

The citation will be updated after publication.

---

## 17. License

Please refer to the license file of this repository. Dataset licenses are governed by their original providers and are not covered by this repository.

---

## 18. Acknowledgements

This project builds on open-source tools and models including Qwen3-VL, BioMedCLIP, PyTorch, Transformers, PEFT, Accelerate, BitsAndBytes, qwen-vl-utils, and open-clip.
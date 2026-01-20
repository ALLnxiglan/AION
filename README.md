# AION: Atomized Inference for Online GMV with Cascaded Non-binary Feedback

AION is an atomized inference method designed for online scenarios, targeting both CVR (Conversion Rate) and GMV (Gross Merchandise Volume) tasks, with adaptation to the business characteristics of cascaded non-binary feedback. This repository provides the implementation code of AION, where the CVR task is fully implemented, and the GMV task is pending—relevant code will be added in subsequent updates.

## Table of Contents

- [Environment Requirements](https://www.doubao.cn)
- [Tasks](https://www.doubao.cn)
- [Acknowledgements](https://www.doubao.cn)

## Environment Requirements

- Operating System: Linux/macOS/Windows (Linux recommended)
- Python Version: 3.8.10
- Dependency Packages (specified versions):

```bash
numpy==1.18.5
tqdm==4.61.2
pandas==1.3.1
scikit-learn==1.0.2  # Fixed typo: scikit_learn → scikit-learn (official PyPI name)
tensorflow==2.4.1
```

### Install Dependencies

Run the following command to install required dependencies quickly:

```bash
pip install numpy==1.18.5 tqdm==4.61.2 pandas==1.3.1 scikit-learn==1.0.2 tensorflow==2.4.1
```

## Tasks

### 1. CVR (Conversion Rate) Task

The CVR task implementation is complete, with support for pretraining and streaming inference. Follow the steps below to run the code:

#### 1.1 Pretrain the Model

First, execute the pretraining step to generate baseline model weights:

```bash
CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
--method Pretrain --mode pretrain \
--model_ckpt_path ./delayed_feedback_release/ckpts/pretrain/pretrain \
--data_path ../data/criteo/data.txt \
--data_cache_path ./delayed_feedback_release/data_cache
```

#### 1.2 Run AION in Streaming Setting

Based on the pretrained model, obtain inference results of the AION method under streaming settings:

```bash
CUDA_VISIBLE_DEVICES=1 python ./src/main.py \
--method AION --mode stream --C 0.5 \
--pretrain_baseline_model_ckpt_path ./delayed_feedback_release/ckpts/pretrain/pretrain \
--data_path ./data/criteo/data.txt \
--data_cache_path ./delayed_feedback_release/data_cache
```

### 2. GMV (Gross Merchandise Volume) Task

The GMV task implementation is currently in progress. This section will be updated with detailed running guidelines, code, and dependencies once the development is completed.

## Acknowledgements

Part of the code in this project is referenced from the DEFUSE repository:   [ychen216/DEFUSE: code of our WWW 2022 paper Asymptotically Unbiased Estimation for Delayed Feedback Modeling via Label Correction](https://github.com/ychen216/DEFUSE/tree/master)
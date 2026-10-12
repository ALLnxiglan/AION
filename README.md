# Beyond Single Delay: Debiased Online GMV Prediction under Cascaded Non-binary Delayed Feedback

This repository provides the implementation of **AION** (Atomic Inference for Online GMV with cascaded Non-binary feedback) for online GMV (Gross Merchandise Volume) prediction under cascaded non-binary delayed feedback. It also includes an adaptation of AION's delayed-feedback module for CVR (Conversion Rate) prediction.

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

The GMV task implementation is complete, with support for pretraining and streaming inference. Follow the steps below to run the code:

#### 2.1 Pretrain the Model

First, execute the pretraining step to generate baseline model weights:

```bash
cd GMV
python method_main.py --model='AION' --mode='pretrain' --gpu_id='0' \
    --pretrain_remarks='AION' --lr=0.001 --epochs=20
```

#### 2.2 Run AION in Streaming Setting

Based on the pretrained model, obtain inference results of the AION method under streaming settings:

```bash
python method_main.py --model='AION' --mode='stream' --gpu_id='0' \
    --pretrain_remarks='AION' --lr=0.003 \
    --boost_weight=0.1 --ga_loss_weight=0.5 \
    --train_start_day=57 --train_end_day=81.875
```

For more details on the GMV task (model architecture, loss function, metrics, etc.), see [GMV/README.md](GMV/README.md).

## Acknowledgements

Part of the code in this project is referenced from the DEFUSE repository: [ychen216/DEFUSE](https://github.com/ychen216/DEFUSE/tree/master)

The GMV task builds upon the OnlineGMV repository: [alimama-tech/OnlineGMV](https://github.com/alimama-tech/OnlineGMV) — which provides the code and benchmark for the WWW 2026 paper *Delayed Feedback Modeling for Post-Click Gross Merchandise Volume Prediction: Benchmark, Insights and Approaches*. We thank the authors for providing the dataset and baseline code.


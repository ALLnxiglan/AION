# AION: Attribution-aware Incremental Online Nudge

A count-price factorization model with sparse Mixture-of-Experts (MoE) for **delayed feedback GMV prediction** in online advertising.

## Overview

In online advertising, GMV (Gross Merchandise Volume) prediction suffers from **delayed feedback**: at inference time, only partial payment observations are available, yet we need to predict the final total GMV over a 7-day attribution window. AION addresses this by:

1. **Count-Price Decomposition**: Predicts GMV as `E[count] × E[price]`, where count is a classification task (1..10 purchases) and price is modeled by a sparse MoE tower with hierarchical gating.
2. **Importance Weighting**: Uses the pretrained model's count posterior to compute `P(Y=N|X) / P(Y>=N|X)`, upweighting later observations that are more informative for the final GMV.
3. **Consistent Train-Test Target**: The main loss targets `final_gmv` (total GMV over the attribution window), ensuring training and evaluation are aligned.

## Model Architecture

```
Input (22 categorical features)
  ├── Count Tower → logits [B, 10] → E[count] ∈ [1, 10]
  └── Price MoE Tower
        ├── Group-level gate → G groups
        ├── Within-group gate → H experts per group
        ├── Top-k sparse routing (k=2)
        └── Expert networks → log-price predictions [B, E]
                                        ↓
                              E[price] = Σ w_e · exp(μ_e)
                                        ↓
                              GMV = E[count] × E[price]
```

- **Count Tower**: 6-layer MLP with Dice activations, outputs 10-class logits.
- **Price MoE Tower**: Hierarchical 2-level gating (group × within-group) with top-k=2 sparse routing. Each expert is a 4-layer MLP with Dice activations predicting in log-price space.
- **Dice Activation**: Data-dependent activation `p·x + (1-p)·α·x` where `p = sigmoid(LayerNorm(x))`.

## Training

### Pretrain

```bash
python method_main.py --model='AION' --mode='pretrain' --gpu_id='0' \
    --pretrain_remarks='AION' --lr=0.001 --epochs=20
```

### Stream Fine-tuning

```bash
python method_main.py --model='AION' --mode='stream' --gpu_id='0' \
    --pretrain_remarks='AION' --lr=0.003 \
    --boost_weight=0.1 --ga_loss_weight=0.5 \
    --train_start_day=57 --train_end_day=81.875
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 0.003 | Learning rate |
| `--boost_weight` | 0.1 | Price supervision loss weight |
| `--ga_loss_weight` | 0.5 | Count CE loss weight |
| `--train_start_day` | 57 | Stream training start day |
| `--train_end_day` | 81.875 | Stream training end day |
| `--stream_step` | 0.125 | Time step per stream chunk (days) |
| `--batch_size` | 1024 | Batch size |
| `--epochs` | 1 | Epochs per stream chunk |

## Loss Function

```
loss = (E[count] · (w_main·L_main + w_price·L_price + w_count·L_count_ce) · iw).mean()
     + w_gate · L_gate
```

| Component | Formula | Weight | Role |
|-----------|---------|--------|------|
| `L_main` | `|log1p(pred) - log1p(final_gmv)|` | 1.0 | Main GMV prediction loss |
| `L_price` | `Σ_e w_e · SmoothL1(μ_e, log(avg_price))` | 0.1 | Per-expert price supervision |
| `L_count_ce` | `CrossEntropy(count_logits, count_label)` | 0.5 | Count classification |
| `L_gate` | `-H(w)` (neg entropy) | 0.01 | Gate diversity regularization |
| `iw` | `P(Y=N|X) / P(Y>=N|X)` | — | Importance weight from pretrained model |

## Metrics

- **AUC**: Area Under ROC Curve for GMV ordering
- **ACC**: Fraction of samples with relative error ≤ 20%
- **ALPR** (↓ lower is better): Average Log-Price Ratio `mean(|log2(pred/true)|)`

## Project Structure

```
├── method_main.py              # Entry point
├── method_src/
│   ├── online_stream.py        # Stream training dispatcher
│   ├── stream_trainer_aion.py  # AION stream trainer
│   ├── data.py                 # Data loading & preprocessing
│   ├── dataset.py              # PyTorch Dataset/DataLoader
│   ├── metrics.py              # Evaluation metrics (AUC, ACC, ALPR, etc.)
│   ├── utils.py                # Logger, seed, early stopping
│   ├── pretrain.py             # Pretrain dispatcher (classifier/calibrator)
│   ├── trainer.py              # Pretrain trainer (classifier/calibrator)
│   └── models/
│       ├── Classifier.py       # Multi-pay classifier
│       └── Calibrator.py       # GMV calibrator with time
├── src/
│   ├── models/
│   │   └── aion.py             # AION model (Count + PriceMoE)
│   ├── pretrain.py             # AION pretrain entry
│   └── trainer.py              # AION pretrain trainer
└── requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- numpy, pandas, scikit-learn, tqdm



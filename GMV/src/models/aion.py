"""
AION: Attribution-aware Incremental Online Nudge

A count-price factorization model with sparse MoE price tower
for delayed feedback GMV prediction in online advertising.

Model output:
    count_logits: [B, 10]  - logits for purchase count classes 1..10
    price_mu:     [B, E]   - log-space price prediction from each expert
    gate_logits:  [B, G+G*H] - raw gating logits (for monitoring)
    w:            [B, E]   - gating weights (softmax, top-k sparse)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocab sizes for 22 categorical features (multi-pay users only)
CATE_BIN_SIZE = (5, 3604778, 1951876, 432750, 14, 19, 2638598, 1962400,
                 11123, 397159, 578622, 3, 622, 107, 8009, 30685, 104, 10, 3, 2, 9, 7)


class Dice(nn.Module):
    """Data-Dependent Activation: p * x + (1-p) * alpha * x, where p = sigmoid(LN(x))."""

    def __init__(self, units, eps=1e-4):
        super().__init__()
        self.bn = nn.LayerNorm(units, eps=eps, elementwise_affine=False)
        self.alpha = nn.Parameter(torch.ones(1, units) * -0.25)

    def forward(self, x):
        normed = self.bn(x)
        p = torch.sigmoid(normed)
        return p * x + (1 - p) * self.alpha * x


class PriceMoe(nn.Module):
    """Sparse Mixture-of-Experts price tower with hierarchical gating.

    Two-level gating: group-level softmax * within-group softmax => expert weights.
    Top-k sparse routing for efficient inference.
    """

    def __init__(self, args, groups=3, experts_per_group=2, gate_topk=2):
        super().__init__()
        self.gate_topk = gate_topk
        self.G = groups
        self.H = experts_per_group
        self.E = self.G * self.H

        self.cate_features = len(CATE_BIN_SIZE)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=bucket_size, embedding_dim=args.embed_dim)
            for bucket_size in CATE_BIN_SIZE
        ])

        input_dim = args.embed_dim * self.cate_features
        self.fc1 = nn.Linear(input_dim, 256); self.d1 = Dice(256)
        self.fc2 = nn.Linear(256, 128);       self.d2 = Dice(128)
        self.fc3 = nn.Linear(128, 64);        self.d3 = Dice(64)

        # Hierarchical gating
        self.gate_g = nn.Linear(64, self.G)           # [B, G]
        self.gate_h = nn.Linear(64, self.G * self.H)  # [B, G*H]

        # Expert networks: each predicts a scalar in log-price space
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32), Dice(32),
                nn.Linear(32, 16), Dice(16),
                nn.Linear(16, 8),  Dice(8),
                nn.Linear(8, 1)
            ) for _ in range(self.E)
        ])

    def forward(self, x):
        x = x.long()
        emb = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(emb, dim=-1)

        x = self.d1(self.fc1(x))
        x = self.d2(self.fc2(x))
        h = self.d3(self.fc3(x))  # [B, 64]

        # Hierarchical gating weights
        logit_g = self.gate_g(h)                              # [B, G]
        w_g = F.softmax(logit_g, dim=1)
        logit_h = self.gate_h(h).view(-1, self.G, self.H)    # [B, G, H]
        w_h = F.softmax(logit_h, dim=2)
        w = (w_g.unsqueeze(2) * w_h).view(-1, self.E)        # [B, E]

        # Top-k sparse routing
        k = self.gate_topk
        if isinstance(k, int) and 0 < k < self.E:
            topk = torch.topk(w, k, dim=1)
            topk_idx = topk.indices

            mask = torch.zeros_like(w).scatter(1, topk_idx, 1.0)
            w = w * mask
            w = w / (w.sum(dim=1, keepdim=True).clamp_min(1e-12))
            w = w.clamp_min(1e-8)

            # Sparse expert forward: only compute routed experts
            B = h.size(0)
            device = h.device
            dtype = h.dtype
            price_mu = torch.zeros(B, self.E, device=device, dtype=dtype)

            for e in range(self.E):
                pos = (topk_idx == e).nonzero(as_tuple=False)
                if pos.numel() == 0:
                    continue
                b_idx = pos[:, 0]
                out = self.experts[e](h.index_select(0, b_idx)).squeeze(1)
                price_mu[b_idx, e] = out
        else:
            # Dense: compute all experts
            w = w.clamp_min(1e-8)
            mus = [expert(h) for expert in self.experts]
            price_mu = torch.cat(mus, dim=1)  # [B, E]

        gate_logits = torch.cat([logit_g, logit_h.view(-1, self.G * self.H)], dim=1)
        return price_mu, gate_logits, w


class Count(nn.Module):
    """Count classification tower: predicts purchase count in {1, ..., 10}."""

    def __init__(self, args):
        super().__init__()
        self.cate_features = len(CATE_BIN_SIZE)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=bucket_size, embedding_dim=args.embed_dim)
            for bucket_size in CATE_BIN_SIZE
        ])
        input_dim = args.embed_dim * self.cate_features

        self.fc1 = nn.Linear(input_dim, 1024);  self.dice1 = Dice(1024)
        self.fc2 = nn.Linear(1024, 512);        self.dice2 = Dice(512)
        self.fc3 = nn.Linear(512, 256);         self.dice3 = Dice(256)
        self.fc4 = nn.Linear(256, 128);         self.dice4 = Dice(128)
        self.fc5 = nn.Linear(128, 64);          self.dice5 = Dice(64)
        self.fc6 = nn.Linear(64, 10)

    def forward(self, x):
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)
        x = self.dice1(self.fc1(x_embed))
        x = self.dice2(self.fc2(x))
        x = self.dice3(self.fc3(x))
        x = self.dice4(self.fc4(x))
        x = self.dice5(self.fc5(x))
        x = self.fc6(x)
        return x


class AION(nn.Module):
    """AION: Attribution-aware Incremental Online Nudge.

    Decomposes GMV prediction into:
        GMV = E[count] * E[price]
    where count is predicted by a classification tower (1..10),
    and price by a sparse MoE tower with hierarchical gating.
    """

    def __init__(self, args, groups=3):
        super().__init__()
        self.groups = groups
        self.count_tower = Count(args)
        self.price_tower = PriceMoe(args, groups=groups)

    def forward(self, x):
        count_logits = self.count_tower(x)
        price_mu, gate_logits, w = self.price_tower(x)
        return count_logits, price_mu, gate_logits, w


from logging import getLogger
import os
import torch
import copy
from tqdm import tqdm
import torch.nn.functional as F
from method_src.metrics import *
import time


class StreamTrainerAion(object):

    def __init__(self, args, pretrained_model, device, train_dataloader, test_dataloader):
        self.args = args
        self.device = device
        self.logger = getLogger()
        self.epochs = args.epochs

        self.learning_rate = args.lr
        self.weight_decay = args.weight_decay
        self.stopping_step = args.stopping_step
        self.learning_rate_scheduler = args.learning_rate_scheduler

        self.model_name = args.model
        self.mode = args.mode
        self.pretrained_model = pretrained_model
        self.model = copy.deepcopy(pretrained_model)
        self.model.to(self.device)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = self._get_optimizer()
        fac = lambda epoch: self.learning_rate_scheduler[0] ** (epoch / self.learning_rate_scheduler[1])
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.loss_gate_weight = 0.01
        self.expected_count = 1.7245  # mean count on pretrain set

    # ------------------------------------------------------------------ #
    #  Helper methods
    # ------------------------------------------------------------------ #

    def _importance_weight(self, features, count_true, clip_max=10.0, normalize=True):
        """Compute importance weight: P(Y=N|X) / P(Y>=N|X) from pretrained count posterior."""
        with torch.no_grad():
            self.pretrained_model.eval()
            pre_count_logits, _, _, _ = self.pretrained_model(features)
            prob = F.softmax(pre_count_logits, dim=1)  # [B, 10], index 0..9 -> count 1..10

            n_idx = (count_true.long() - 1).clamp(0, 9)   # [B]
            n_idx_u = n_idx.unsqueeze(1)                     # [B, 1]

            # Numerator: P(Y=N|X)
            numer = prob.gather(1, n_idx_u).squeeze(1)      # [B]

            # Denominator: P(Y>=N|X) via reverse cumulative sum
            prob_rev = torch.flip(prob, dims=[1])
            cumsum_rev = torch.cumsum(prob_rev, dim=1)
            cdf_ge = torch.flip(cumsum_rev, dims=[1])       # [B, 10], col j = P(Y>=j+1)
            denom = cdf_ge.gather(1, n_idx_u).squeeze(1)    # [B]

            w = numer / denom.clamp_min(1e-6)               # [B]
            if clip_max is not None:
                w = w.clamp_max(clip_max)
            if normalize:
                w = w / w.mean().clamp_min(1e-6)
        return w

    def _gating_entropy_loss(self, w):
        """Negative entropy of gating weights (minimize => maximize entropy)."""
        ent = -(w.clamp_min(1e-12).log() * w).sum(dim=1).mean()
        return -ent

    def _count_expected(self, logits):
        """Compute expected count from logits: E[count] = sum(k * P(count=k)) for k=1..10."""
        prob = F.softmax(logits, dim=1)  # [B, 10]
        classes = torch.arange(1, 11, device=logits.device, dtype=prob.dtype).view(1, -1)
        return (prob * classes).sum(dim=1)  # [B]

    def _get_optimizer(self):
        params = list(self.model.parameters())
        if self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")
        return optimizer

    # ------------------------------------------------------------------ #
    #  Test
    # ------------------------------------------------------------------ #

    def test(self, day):
        self.model.eval()
        self.pretrained_model.eval()

        all_gmv_preds = []
        pretrain_all_gmv_preds = []
        all_gmv_labels = []
        multi_tag = []
        metrics_dict = {}

        day_loader = self.test_dataloader.get_day_dataloader(day)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Testing Day {day + 1}", leave=False)
        self.logger.info(f"Testing Day {day + 1}...")

        # Inference timing and GPU memory stats
        use_cuda = (self.device is not None) and torch.cuda.is_available() and ("cuda" in str(self.device))
        infer_time_total = 0.0
        infer_batches = 0
        infer_samples = 0

        if use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_gmv_label = batch['final_gmv'].to(self.device)

                if use_cuda:
                    torch.cuda.synchronize(self.device)
                t0 = time.perf_counter()

                # Model prediction: count_pred * E_price
                count_logits, price_mu, gate_logits, w = self.model(features)
                count_pred = self._count_expected(count_logits).clamp(1.0, 10.0)
                price_mu = price_mu.clamp(1e-12, 20.0)
                E_price = (w * torch.exp(price_mu)).sum(dim=1)
                pgmv = count_pred * E_price

                # Pretrained baseline
                pre_count_logits, pre_price_mu, pre_gate_logits, pre_w = self.pretrained_model(features)
                pre_count_pred = self._count_expected(pre_count_logits).clamp(1.0, 10.0)
                pre_price_mu = pre_price_mu.clamp(1e-12, 20.0)
                pre_E_price = (pre_w * torch.exp(pre_price_mu)).sum(dim=1)
                pretrain_pgmv = pre_count_pred * pre_E_price

                if use_cuda:
                    torch.cuda.synchronize(self.device)
                t1 = time.perf_counter()
                infer_time_total += (t1 - t0)
                infer_batches += 1
                infer_samples += features.size(0)

                all_gmv_preds.append(pgmv.view(-1).cpu())
                pretrain_all_gmv_preds.append(pretrain_pgmv.view(-1).cpu())
                all_gmv_labels.append(pay_gmv_label.cpu())
                multi_tag.append(batch['multi_tag'].cpu())

        all_gmv_preds = torch.cat(all_gmv_preds, dim=0).numpy()
        pretrain_all_gmv_preds = torch.cat(pretrain_all_gmv_preds, dim=0).numpy()
        all_gmv_labels = torch.cat(all_gmv_labels, dim=0).numpy()
        multi_tag = torch.cat(multi_tag, dim=0).numpy()

        test_acc = get_acc(all_gmv_preds, all_gmv_labels)
        test_auc = get_auc(all_gmv_preds, all_gmv_labels)
        test_alpr = get_alpr(all_gmv_preds, all_gmv_labels)

        pretrain_test_acc = get_acc(pretrain_all_gmv_preds, all_gmv_labels)
        pretrain_test_auc = get_auc(pretrain_all_gmv_preds, all_gmv_labels)
        pretrain_test_alpr = get_alpr(pretrain_all_gmv_preds, all_gmv_labels)

        test_result = f"Day {day + 1} - Test AUC: {test_auc:.4f}, Test ACC: {test_acc:.4f}, Test ALPR: {test_alpr:.4f}\n"
        test_result += f"Pretrain Test AUC: {pretrain_test_auc:.4f}, Pretrain Test ACC: {pretrain_test_acc:.4f}, Pretrain Test ALPR:{pretrain_test_alpr:.4f}\n"

        # Inference timing summary
        if infer_batches > 0:
            avg_ms_per_batch = (infer_time_total / infer_batches) * 1000.0
            metrics_dict["infer_avg_ms_per_batch"] = avg_ms_per_batch
            test_result += f"Infer avg time: {avg_ms_per_batch:.3f} ms/batch\n"

        if infer_samples > 0:
            avg_ms_per_sample = (infer_time_total / infer_samples) * 1000.0
            metrics_dict["infer_avg_ms_per_sample"] = avg_ms_per_sample
            test_result += f"Infer avg time: {avg_ms_per_sample:.6f} ms/sample\n"

        if use_cuda:
            peak_alloc = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            peak_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
            cur_alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            cur_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)

            metrics_dict["gpu_peak_mem_alloc_mb"] = peak_alloc
            metrics_dict["gpu_peak_mem_reserved_mb"] = peak_reserved
            metrics_dict["gpu_cur_mem_alloc_mb"] = cur_alloc
            metrics_dict["gpu_cur_mem_reserved_mb"] = cur_reserved

            test_result += (
                f"GPU mem (MB) peak_alloc={peak_alloc:.1f}, peak_reserved={peak_reserved:.1f}, "
                f"cur_alloc={cur_alloc:.1f}, cur_reserved={cur_reserved:.1f}\n"
            )

        self.logger.info(test_result)

        metrics_dict['test_auc'] = test_auc
        metrics_dict['test_acc'] = test_acc
        metrics_dict['test_alpr'] = test_alpr
        metrics_dict['pretrain_test_auc'] = pretrain_test_auc
        metrics_dict['pretrain_test_acc'] = pretrain_test_acc
        metrics_dict['pretrain_test_alpr'] = pretrain_test_alpr

        return metrics_dict

    # ------------------------------------------------------------------ #
    #  Train
    # ------------------------------------------------------------------ #

    def train(self):
        all_day_metrics = []

        for day in tqdm(range(len(self.train_dataloader)), desc="Days"):
            for epoch_idx in tqdm(range(self.epochs), desc="Epochs", leave=False):
                self.model.train()
                total_loss = 0.0
                day_loader = self.train_dataloader.get_day_dataloader(day)
                tqdm_day_dataloader = tqdm(day_loader, desc=f"Training Day {day} - Epoch {epoch_idx + 1}", leave=False)
                for batch_idx, batch in enumerate(tqdm_day_dataloader):
                    features = batch['features'].to(self.device)
                    now_pay_gmv = batch['first_pay_gmv'].to(self.device)   # current cumulative GMV
                    final_gmv = batch['final_gmv'].to(self.device)          # final total GMV
                    now_count = batch['now_count'].to(self.device)

                    count_logits, price_mu, gate_logits, w = self.model(features)
                    gmv_now = now_pay_gmv.view(-1).float()                  # [B] for price supervision
                    gmv_final = final_gmv.view(-1).float()                  # [B] for main loss target
                    count_true = now_count.view(-1).float().clamp(1.0, 10.0) # [B]

                    iw = self._importance_weight(features, count_true, clip_max=10.0, normalize=True)
                    self.optimizer.zero_grad()

                    # 1) Count classification (label 1..10 -> index 0..9)
                    y_cnt = (count_true.long() - 1).clamp(0, 9)
                    loss_count_ce = F.cross_entropy(count_logits, y_cnt, reduction='none')
                    count_pred = self._count_expected(count_logits).clamp(1.0, 10.0)

                    # 2) Price supervision (log space, using observed average price)
                    price_mu = price_mu.clamp(1e-12, 20.0)
                    price_tag = (gmv_now / (count_true + 1e-6)).clamp(min=1e-12)
                    y_log = torch.log(price_tag)
                    per_exp_loss = F.smooth_l1_loss(price_mu, y_log.unsqueeze(1),
                                                    beta=1.0, reduction='none')
                    loss_price = (w * per_exp_loss).sum(dim=1)

                    # 3) Main GMV loss (target = final_gmv, consistent with test)
                    E_price = (w * torch.exp(price_mu)).sum(dim=1)
                    pamt = count_pred * E_price
                    loss_main = F.l1_loss(torch.log1p(pamt), torch.log1p(gmv_final), reduction='none')

                    if torch.isnan(loss_main).any() or torch.isinf(loss_main).any():
                        self.logger.warning(f"NaN/Inf in loss_main: pamt min/max={pamt.min().item():.4f}/{pamt.max().item():.4f}, gmv_final min/max={gmv_final.min().item():.4f}/{gmv_final.max().item():.4f}")
                        loss_main = torch.clamp(loss_main, min=0, max=1e6)

                    # 4) Gating entropy regularization
                    loss_gate = self.loss_gate_weight * self._gating_entropy_loss(w)

                    # 5) Total loss
                    #    loss = (expected_count * (w_main*loss_main + w_price*loss_price + w_count*loss_count_ce) * iw).mean()
                    #         + w_gate * loss_gate
                    w_main = 1.0
                    w_price = self.args.boost_weight   # default 0.1
                    w_count = self.args.ga_loss_weight  # default 0.5
                    w_gate = 0.01
                    loss = (self.expected_count * (w_main*loss_main + w_price*loss_price + w_count*loss_count_ce) * iw).mean() \
                         + w_gate * loss_gate

                    # Numerical guard
                    if not torch.isfinite(loss_main.mean()) or not torch.isfinite(loss_price.mean()):
                        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        self.logger.warning(
                            f"Non-finite: mu[min,max]={price_mu.min().item():.2f}/{price_mu.max().item():.2f}, "
                            f"E_price[min,max]={E_price.min().item():.2f}/{E_price.max().item():.2f}"
                        )
                        continue

                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                self.lr_scheduler.step()
                avg_loss = total_loss / len(day_loader)
                self.logger.info(f"Day {day} - Epoch {epoch_idx + 1} - Avg Loss: {avg_loss:.4f}")

            # Test after each day
            metrics_dict = self.test(day)
            all_day_metrics.append(metrics_dict)

        res = f'============lr: {self.args.lr}============\n'
        self.logger.info("Training completed for all days.")
        avg_metrics = self.aggregate_metrics(all_day_metrics)
        for k, v in avg_metrics.items():
            self.logger.info(f"Average {k}: {v:.4f}")
            res += f"Average {k}: {v:.4f}\n"
        res += '========================================\n'
        return res

    def aggregate_metrics(self, metrics_list):
        total = {}
        for key in metrics_list[0].keys():
            total[key] = 0.0
        for daily_metrics in metrics_list:
            for key, value in daily_metrics.items():
                total[key] += value
        for key in total:
            total[key] /= len(metrics_list)
        return total

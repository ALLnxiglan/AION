from logging import getLogger
import os
import json
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from src.metrics import get_alpr, get_acc, get_auc


class PretrainedTrainer(object):
    """
    DFM-only trainer/tester + in-test case study mining.

    Model forward must return:
        count_logits: [B,10]
        price_mu:     [B,E]   (log-price space)
        gate_logits:  unused here
        w:            [B,E]
    """

    def __init__(self, args, model, device):
        self.args = args
        self.logger = getLogger()

        self.device = device
        self.epochs = args.epochs

        self.learning_rate = args.lr
        self.weight_decay = args.weight_decay
        self.learning_rate_scheduler = args.learning_rate_scheduler

        self.model_name = args.model
        self.mode = args.mode
        self.model_pth = os.path.join(
            args.model_save_pth,
            f"{self.model_name}_{self.mode}_{args.seed}_{args.pretrain_remarks}.pth"
        )

        self.model = model.to(self.device)

        # gating entropy regularization weight (training)
        self.loss_gate_weight = 0.01

        # case study config (testing)
        self.case_enabled = True
        self.case_count_pair = (1, 4)      # want one with count=1 and one with count=4 in same batch
        self.case_gmv_log_eps = 0.03       # log1p(gmv) distance threshold for "similar GMV"
        self.case_count_tol = 0.5          # abs(pred_count - true_count) <= tol
        self.case_price_rel_tol = 0.20     # abs(pred_price - true_price)/true_price <= tol
        self.case_min_gmv = 0.0            # set e.g. 1000.0 if you also want gmv>1000
        self.case_save_path = "study_case_pair.json"
        self.case_found = False

        self.optimizer = self._get_optimizer()
        fac = lambda epoch: self.learning_rate_scheduler[0] ** (epoch / self.learning_rate_scheduler[1])
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)

    # ---------------- utils ----------------
    def _get_optimizer(self):
        if self.args.optimizer == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.args.optimizer == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")

    def _count_expected(self, logits):
        # logits: [B,10] (classes 1..10)
        prob = F.softmax(logits, dim=1)
        classes = torch.arange(1, 11, device=logits.device, dtype=prob.dtype).view(1, -1)
        return (prob * classes).sum(dim=1)  # [B]

    def _gating_entropy_loss(self, w):
        ent = -(w.clamp_min(1e-12).log() * w).sum(dim=1).mean()
        return -ent

    def _try_find_case_in_batch(
        self,
        features,                 # [B,F] long
        gmv_true,                 # [B] float
        count_true,               # [B] int/float
        count_logits, price_mu, w,
        batch_idx,
        day=None,
    ):
        """
        Find one pair within this batch:
          - true count equals (1,4)
          - GMV similar (log1p diff <= eps)
          - predictions are "accurate" for both samples:
              abs(count_pred-count_true) <= count_tol
              rel_err(price_pred, price_true) <= price_rel_tol
        """
        c1, c4 = self.case_count_pair

        gmv_true = gmv_true.view(-1).float()
        ct = count_true.view(-1).float().clamp(1.0, 10.0)

        # optional GMV floor
        valid_gmv = gmv_true >= float(self.case_min_gmv)

        # true avg price
        pt = (gmv_true / (ct + 1e-6)).clamp_min(1e-12)

        # preds
        count_pred = self._count_expected(count_logits).clamp(1.0, 10.0)
        price_mu = price_mu.clamp(1e-12, 20.0)
        price_pred = (w * torch.exp(price_mu)).sum(dim=1)  # E_price

        # "accurate" constraints
        ok_count = (count_pred - ct).abs() <= float(self.case_count_tol)
        ok_price = ((price_pred - pt).abs() / (pt + 1e-6)) <= float(self.case_price_rel_tol)
        ok = ok_count & ok_price & valid_gmv

        # candidate indices in this batch
        idx1 = torch.where((ct == float(c1)) & ok)[0]
        idx4 = torch.where((ct == float(c4)) & ok)[0]
        if idx1.numel() == 0 or idx4.numel() == 0:
            return None

        log_g = torch.log1p(gmv_true.clamp_min(0.0))
        best = None

        # choose the closest GMV (log space) pair
        idx4_log = log_g[idx4]
        for i in idx1.tolist():
            diffs = (idx4_log - log_g[i]).abs()
            jpos = int(torch.argmin(diffs).item())
            j = int(idx4[jpos].item())
            gdiff = float(diffs[jpos].item())
            if gdiff <= float(self.case_gmv_log_eps):
                score = -gdiff
                if best is None or score > best["score"]:
                    best = {"i": int(i), "j": int(j), "gdiff": gdiff, "score": score}

        if best is None:
            return None

        i, j = best["i"], best["j"]

        # top experts (for interpretability)
        k = min(3, w.size(1))
        topi_i = torch.topk(w[i], k=k).indices.detach().cpu().tolist()
        topv_i = torch.topk(w[i], k=k).values.detach().cpu().tolist()
        topi_j = torch.topk(w[j], k=k).indices.detach().cpu().tolist()
        topv_j = torch.topk(w[j], k=k).values.detach().cpu().tolist()

        case = {
            "day": day,
            "batch_idx": int(batch_idx),
            "case_count_pair": [int(c1), int(c4)],
            "gmv_log_eps": float(self.case_gmv_log_eps),
            "count_tol": float(self.case_count_tol),
            "price_rel_tol": float(self.case_price_rel_tol),
            "min_gmv": float(self.case_min_gmv),

            "i": int(i),
            "j": int(j),
            "gmv_logdiff": float(best["gdiff"]),

            # sample A (count=1)
            "A": {
                "true_gmv": float(gmv_true[i].item()),
                "true_count": float(ct[i].item()),
                "true_price": float(pt[i].item()),
                "pred_count": float(count_pred[i].item()),
                "pred_price": float(price_pred[i].item()),
                "pred_gmv": float((count_pred[i] * price_pred[i]).item()),
                "count_abs_err": float((count_pred[i] - ct[i]).abs().item()),
                "price_rel_err": float(((price_pred[i] - pt[i]).abs() / (pt[i] + 1e-6)).item()),
                "top_expert_ids": topi_i,
                "top_expert_w": topv_i,
                "feature_ids": features[i].detach().cpu().tolist(),
            },

            # sample B (count=4)
            "B": {
                "true_gmv": float(gmv_true[j].item()),
                "true_count": float(ct[j].item()),
                "true_price": float(pt[j].item()),
                "pred_count": float(count_pred[j].item()),
                "pred_price": float(price_pred[j].item()),
                "pred_gmv": float((count_pred[j] * price_pred[j]).item()),
                "count_abs_err": float((count_pred[j] - ct[j]).abs().item()),
                "price_rel_err": float(((price_pred[j] - pt[j]).abs() / (pt[j] + 1e-6)).item()),
                "top_expert_ids": topi_j,
                "top_expert_w": topv_j,
                "feature_ids": features[j].detach().cpu().tolist(),
            },
        }

        # enforce A is count=1, B is count=4 (swap if needed)
        if case["A"]["true_count"] != float(c1):
            case["A"], case["B"] = case["B"], case["A"]
            case["i"], case["j"] = case["j"], case["i"]

        return case

    # ---------------- eval ----------------
    def test(self, test_loader, day=None):
        """
        If you trained already and go straight to test(), this will:
          - compute metrics
          - ALSO mine one case study pair from one batch (count=1 vs count=4, GMV similar, predictions accurate)
            and save to self.case_save_path
        """
        self.model.eval()

        all_gmv_labels = []
        all_pamt_preds = []
        all_count_preds = []
        all_mgmv_preds = []
        count_labels = []
        mgmv_labels = []
        metrics_dict = {}

        self.logger.info("Testing...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                features = batch["features"].to(self.device)
                pay_gmv_label = batch["final_gmv"].to(self.device).view(-1)
                count_tag_label = batch["total_count"].to(self.device).view(-1)

                # DFM forward
                count_logits, price_mu, gate_logits, w = self.model(features)

                # prediction
                count_pred = self._count_expected(count_logits).clamp(1.0, 10.0)
                price_mu = price_mu.clamp(1e-12, 20.0)
                E_price = (w * torch.exp(price_mu)).sum(dim=1)
                pamt = count_pred * E_price

                # ---- case study mining: find once and save ----
                if self.case_enabled and (not self.case_found):
                    case = self._try_find_case_in_batch(
                        features=features,
                        gmv_true=pay_gmv_label,
                        count_true=count_tag_label,
                        count_logits=count_logits,
                        price_mu=price_mu,
                        w=w,
                        batch_idx=batch_idx,
                        day=day,
                    )
                    if case is not None:
                        self.case_found = True
                        self.logger.info("✅ Found study case pair (within one batch) and predictions are accurate.")
                        self.logger.info(json.dumps(case, ensure_ascii=False, indent=2))
                        with open(self.case_save_path, "w") as f:
                            json.dump(case, f, ensure_ascii=False, indent=2)
                        self.logger.info(f"Saved case to: {self.case_save_path}")

                # record for metrics
                all_pamt_preds.append(pamt.detach().cpu())
                all_gmv_labels.append(pay_gmv_label.detach().cpu())

                all_count_preds.append(count_pred.detach().cpu())
                count_labels.append(count_tag_label.float().detach().cpu())

                price_tag_label = (pay_gmv_label.float() / (count_tag_label.float() + 1e-6)).clamp(min=0.0)
                all_mgmv_preds.append(E_price.detach().cpu())
                mgmv_labels.append(price_tag_label.detach().cpu())

        all_pamt_preds = torch.cat(all_pamt_preds, dim=0).numpy()
        all_gmv_labels = torch.cat(all_gmv_labels, dim=0).numpy()

        # extra stats
        all_count_preds_np = torch.cat(all_count_preds, dim=0).numpy()
        count_labels_np = torch.cat(count_labels, dim=0).numpy()
        count_mae = np.mean(np.abs(all_count_preds_np - count_labels_np))

        all_mgmv_preds_np = torch.cat(all_mgmv_preds, dim=0).numpy()
        mgmv_labels_np = torch.cat(mgmv_labels, dim=0).numpy()
        mgmv_mae = np.mean(np.abs(all_mgmv_preds_np - mgmv_labels_np))

        # overall metrics
        test_alpr = get_alpr(all_pamt_preds, all_gmv_labels)
        test_acc = get_acc(all_pamt_preds, all_gmv_labels)
        test_auc = get_auc(all_pamt_preds, all_gmv_labels)

        self.logger.info(f"Test ALPR: {test_alpr:.4f}")
        self.logger.info(f"Test ACC: {test_acc:.4f}")
        self.logger.info(f"Test AUC: {test_auc:.4f}")
        self.logger.info(f"Test Count MAE: {count_mae:.4f}")
        self.logger.info(f"Test MGMV  MAE: {mgmv_mae:.4f}")
        if self.case_enabled and (not self.case_found):
            self.logger.info("No study-case pair found under current thresholds. "
                             "Try relaxing: case_gmv_log_eps / case_count_tol / case_price_rel_tol / case_min_gmv.")

        metrics_dict["test_auc"] = test_auc
        metrics_dict["test_alpr"] = test_alpr
        metrics_dict["test_acc"] = test_acc
        metrics_dict["test_count_mae"] = count_mae
        metrics_dict["test_mgmv_mae"] = mgmv_mae
        metrics_dict["case_found"] = bool(self.case_found)
        return metrics_dict

    # ---------------- train ----------------
    def train(self, train_loader, test_loader):
        """
        If a trained model exists, it loads and directly enters test().
        Otherwise trains for epochs, saves, then tests.
        """
        if os.path.isfile(self.model_pth):
            self.logger.info(f"model_pth {self.model_pth} exists.")
            self.logger.info("loading trained model and directly testing...")
            self.model.load_state_dict(torch.load(self.model_pth, map_location=self.device))
            return self.test(test_loader)

        for epoch_idx in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            self.logger.info(f"Epoch {epoch_idx+1}/{self.epochs} training...")

            for batch_idx, batch in enumerate(tqdm(train_loader)):
                features = batch["features"].to(self.device)
                pay_gmv_label = batch["final_gmv"].to(self.device).view(-1)
                count_tag = batch["total_count"].to(self.device).view(-1)

                gmv = pay_gmv_label.float()
                count_true = count_tag.float().clamp(1.0, 10.0)

                self.optimizer.zero_grad()

                count_logits, price_mu, gate_logits, w = self.model(features)

                # count CE
                y_cnt = (count_true.long() - 1).clamp(0, 9)
                loss_count_ce = F.cross_entropy(count_logits, y_cnt, label_smoothing=0.05)

                # price loss (log space)
                price_mu = price_mu.clamp(1e-12, 20.0)
                price_tag = (gmv / (count_true + 1e-6)).clamp(min=1e-12)
                y_log = torch.log(price_tag)
                per_exp_loss = F.smooth_l1_loss(price_mu, y_log.unsqueeze(1), beta=1.0, reduction="none")
                loss_price = (w * per_exp_loss).sum(dim=1).mean()

                # main GMV loss
                count_pred = self._count_expected(count_logits).clamp(1.0, 10.0)
                log_price_hat = (w * price_mu).sum(dim=1)
                E_price = torch.exp(log_price_hat)
                pamt = count_pred * E_price
                loss_main = F.l1_loss(torch.log1p(pamt), torch.log1p(gmv))

                # gating entropy
                loss_gate = self.loss_gate_weight * self._gating_entropy_loss(w)

                loss = loss_main + 0.4 * loss_price + 0.6 * loss_count_ce + loss_gate

                if not torch.isfinite(loss):
                    self.logger.warning("Non-finite loss skipped.")
                    continue

                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item())

            self.lr_scheduler.step()
            avg_loss = total_loss / max(len(train_loader), 1)
            self.logger.info(f"Epoch {epoch_idx+1}/{self.epochs}, Average Loss: {avg_loss:.4f}")

            parent_dir = os.path.dirname(self.model_pth)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            torch.save(self.model.state_dict(), self.model_pth)
            self.logger.info(f"Model saved at Epoch {epoch_idx+1}")

        return self.test(test_loader)

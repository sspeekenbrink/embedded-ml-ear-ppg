from sklearn.metrics import f1_score, classification_report, confusion_matrix, matthews_corrcoef

from itertools import product
from copy import deepcopy
import pandas as pd

from ppg_pipeline.models import build_cnn, CNNConfig, DilatedCNNConfig, build_dilated_cnn
from ppg_pipeline.utils import save_model_bundle

import logging

import torch
from torch import nn

from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

from sklearn.metrics import roc_curve


import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, f1_score

PREDICTION_THRESHOLD = 0.5


def _config_to_dict(cfg):
    """Convert config object to dictionary for JSON serialization and filtering."""
    if isinstance(cfg, CNNConfig):
        return {
            "config_type": "CNNConfig",
            "base_channels": cfg.base_channels,
            "num_blocks": cfg.num_blocks,
            "kernel_size": cfg.kernel_size,
            "dilation": cfg.dilation,
            "pool_every": cfg.pool_every,
            "use_batchnorm": cfg.use_batchnorm,
            "activation": cfg.activation,
            "pooling": cfg.pooling,
            "double_every": cfg.double_every
        }
    elif isinstance(cfg, DilatedCNNConfig):
        return {
            "config_type": "DilatedCNNConfig",
            "in_channels": cfg.in_channels,
            "enc1_channels": cfg.enc1_channels,
            "enc1_kernel": cfg.enc1_kernel,
            "enc2_channels": cfg.enc2_channels,
            "enc2_kernel": cfg.enc2_kernel,
            "dec1_channels": cfg.dec1_channels,
            "dec1_kernel": cfg.dec1_kernel,
            "dec2_channels": cfg.dec2_channels,
            "dec2_kernel": cfg.dec2_kernel,
            "bottleneck_channels": cfg.bottleneck_channels,
            "num_dilated_layers": cfg.num_dilated_layers,
            "dilated_kernel": cfg.dilated_kernel,
            "base_dilation": cfg.base_dilation,
            "out_channels": cfg.out_channels,
            "activation": cfg.activation,
            "pooling": cfg.pooling
        }
    else:
        # Fallback for unknown config types
        return {"config_type": str(type(cfg).__name__), "config_str": str(cfg)}


def _match_length(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Crops the longer tensor to match the length of the shorter one along the last dimension."""
    L = min(a.size(-1), b.size(-1))
    return a[..., :L], b[..., :L]




@torch.no_grad()
def _collect_logits_and_targets(model, loader, device):
    model.eval()
    logits_all, targets_all = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        logits = model(X)
        logits, y = _match_length(logits, y)
        logits_all.append(logits.detach().float().cpu())
        targets_all.append(y.detach().float().cpu())
    logits_all = torch.cat(logits_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    return logits_all, targets_all

def _best_thresh_for_f1_pos_from_logits(logits_cpu, targets_cpu):
    """
    Returns threshold t* maximizing F1 of positive class on given logits/targets.
    """
    probs = torch.sigmoid(logits_cpu).numpy().ravel()
    y_true = targets_cpu.numpy().astype(np.int32).ravel()

    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    if thresholds.size == 0:
        return 0.5, 0.0
    idx = int(np.nanargmax(f1[:-1]))  # align with thresholds
    return float(thresholds[idx]), float(f1[idx])


def _best_thresh_for_mcc_from_logits(logits_cpu, targets_cpu):
    """
    Returns threshold t* maximizing Matthews Correlation Coefficient (MCC)
    on given logits/targets.

    Args:
        logits_cpu (torch.Tensor): Raw logits from the model (CPU).
        targets_cpu (torch.Tensor): Binary ground truth targets (CPU).

    Returns:
        float: The best threshold.
        float: The best MCC score.
    """
    probs = torch.sigmoid(logits_cpu).numpy().ravel()
    y_true = targets_cpu.numpy().astype(np.int32).ravel()

    fpr, tpr, thresholds = roc_curve(y_true, probs)

    P = np.sum(y_true)
    N = len(y_true) - P

    tp = tpr * P
    fp = fpr * N
    tn = (1 - fpr) * N
    fn = (1 - tpr) * P

    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / (denominator + 1e-12)

    if thresholds.size == 0:
        return 0.5, 0.0

    idx = int(np.nanargmax(mcc))

    return float(thresholds[idx]), float(mcc[idx])

def _threshold_from_args_or_tune(args, logits_cpu, targets_cpu):
    """
    1) if --decision_threshold is set: use it
    2) elif --tune_threshold: tune t* for F1(pos)
    3) else: 0.5
    """
    if args.decision_threshold is not None:
        t = float(max(1e-6, min(1.0 - 1e-6, args.decision_threshold)))
        return t, None
    if args.tune_threshold:
        return _best_thresh_for_f1_pos_from_logits(logits_cpu, targets_cpu)
    return 0.5, None

def _metrics_at_threshold(logits_cpu, targets_cpu, threshold):
    """
    Compute metrics at a fixed threshold.
    Returns dict with: f1_pos, f1_macro, precision_pos, recall_pos, acc, mcc
    """
    probs = torch.sigmoid(logits_cpu).numpy().ravel()
    y_true = targets_cpu.numpy().astype(np.int32).ravel()
    y_pred = (probs >= threshold).astype(np.int32)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1], average=None, zero_division=0
    )
    f1_macro = f1_score(y_true, y_pred, average="macro")
    acc = (y_true == y_pred).mean()
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "precision_pos": float(precision[1]),
        "recall_pos": float(recall[1]),
        "f1_pos": float(f1[1]),
        "f1_macro": float(f1_macro),
        "accuracy": float(acc),
        "mcc": float(mcc),
    }

def nms_1d(scores: np.ndarray, threshold: float, window: int) -> np.ndarray:
    """Apply 1D NMS to a single 1D score vector. Returns uint8 mask with 1s at kept peaks."""
    L = scores.shape[0]
    out = np.zeros(L, dtype=np.uint8)

    cand = np.where(scores >= threshold)[0]
    if cand.size == 0:
        return out

    order = np.argsort(scores[cand])[::-1]
    suppressed = np.zeros(L, dtype=bool)

    for o in order:
        i = cand[o]
        if suppressed[i]:
            continue
        out[i] = 1
        left  = max(0, i - window)
        right = min(L, i + window + 1)
        suppressed[left:right] = True

    return out


def apply_nms_over_windows(probs: np.ndarray, threshold: float, window: int) -> np.ndarray:
    if probs.ndim == 3 and probs.shape[1] == 1:
        flat = probs[:, 0, :]
        restore = lambda arr: arr[:, None, :]
    elif probs.ndim == 2:
        flat = probs
        restore = lambda arr: arr
    else:
        L = probs.shape[-1]
        flat = probs.reshape(-1, L)
        new_prefix = probs.shape[:-1]
        restore = lambda arr: arr.reshape(*new_prefix, L)

    out = np.stack(
        [nms_1d(row, threshold=threshold, window=window) for row in flat],
        axis=0
    ).astype(np.uint8)

    return restore(out)


def postprocess_preds(probs: np.ndarray, threshold: float, nms_window: int) -> np.ndarray:
    if nms_window > 0:
        return apply_nms_over_windows(probs, threshold, nms_window)
    return (probs >= threshold).astype(np.uint8)


def _evaluate_with_threshold(model, loader, args, threshold, *, log_prefix="Final Evaluation", print_reports=True):
    device = args.device
    nms_window = int(args.nms)
    model.eval()

    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            logits = model(X)
            logits, y = _match_length(logits, y)
            probs = torch.sigmoid(logits).cpu().numpy()

            preds = postprocess_preds(probs, threshold=threshold, nms_window=nms_window)

            y_true_all.append(y.cpu().numpy().astype(np.uint8))
            y_pred_all.append(preds.astype(np.uint8))

    y_true = np.concatenate(y_true_all, axis=0).ravel()
    y_pred = np.concatenate(y_pred_all, axis=0).ravel()

    if print_reports:
        print(f"\n{log_prefix} (t = {threshold:.3f}, nms = {nms_window})")
        print(classification_report(y_true, y_pred, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        mcc = matthews_corrcoef(y_true, y_pred)
        print(f"MCC: {mcc:.4f}")

    if args.event_tol is not None:
        tol = int(args.event_tol)
        evt = event_prf1(y_true_all, y_pred_all, tol=tol)
        if print_reports:
            tol_disp = int(args.event_tol)
            print(
                f"Event-level (±{tol_disp}): P={evt['precision']:.4f} R={evt['recall']:.4f} F1={evt['f1']:.4f}  "
                f"[TP={evt['tp']} FP={evt['fp']} FN={evt['fn']}]"
            )
    return y_true_all, y_pred_all


def _compute_metrics_from_binary_lists(y_true_list, y_pred_list):
    y_true = np.concatenate(y_true_list, axis=0).ravel()
    y_pred = np.concatenate(y_pred_list, axis=0).ravel()

    f1_0, f1_1 = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "f1_class_0": float(f1_0),
        "f1_class_1": float(f1_1),
        "f1_macro": float(f1_macro),
        "mcc": float(mcc),
        "report": report,
        "confusion_matrix": cm,
    }


def train_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: str) -> float:
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        logits, y = _match_length(logits, y)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: str) -> dict:
    model.eval()
    running_loss = 0.0
    all_preds, all_trues = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        logits, y = _match_length(logits, y)

        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)

        probs = torch.sigmoid(logits)
        all_preds.append((probs >= PREDICTION_THRESHOLD).int().cpu())
        all_trues.append(y.int().cpu())

    preds_flat = torch.cat(all_preds).flatten().numpy()
    trues_flat = torch.cat(all_trues).flatten().numpy()

    f1_0, f1_1 = f1_score(trues_flat, preds_flat, labels=[0, 1], average=None, zero_division=0)
    f1_macro = f1_score(trues_flat, preds_flat, average="macro", zero_division=0)
    mcc = matthews_corrcoef(trues_flat, preds_flat)

    report = classification_report(trues_flat, preds_flat, digits=4, zero_division=0)
    cm = confusion_matrix(trues_flat, preds_flat)

    return {
        "loss": running_loss / len(loader.dataset),
        "f1_class_0": f1_0,
        "f1_class_1": f1_1,
        "f1_macro": f1_macro,
        "mcc": mcc,
        "report": report,
        "confusion_matrix": cm,
    }


def run_single_training(model, train_loader, val_loader, criterion, optimizer, args):
    logging.info("Starting single model training...")

    training_metrics = {
        "epochs": [],
        "train_losses": [],
        "val_losses": [],
        "val_f1_class_0": [],
        "val_f1_class_1": [],
        "val_f1_macro": [],
        "val_mcc": [],
        "val_precision_pos": [],
        "val_recall_pos": []
    }

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
        val_metrics = evaluate(model, val_loader, criterion, args.device)
        
        training_metrics["epochs"].append(epoch)
        training_metrics["train_losses"].append(float(train_loss))
        training_metrics["val_losses"].append(float(val_metrics['loss']))
        training_metrics["val_f1_class_0"].append(float(val_metrics['f1_class_0']))
        training_metrics["val_f1_class_1"].append(float(val_metrics['f1_class_1']))
        training_metrics["val_f1_macro"].append(float(val_metrics['f1_macro']))
        training_metrics["val_mcc"].append(float(val_metrics['mcc']))
        
        logging.info(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1 (Macro): {val_metrics['f1_macro']:.4f} | "
            f"Val MCC: {val_metrics['mcc']:.4f}"
        )

    if args.decision_threshold is not None:
        best_threshold = float(args.decision_threshold)
        logging.info(f"Using fixed decision threshold from CLI: t = {best_threshold:.3f}")
    elif args.tune_threshold:
        logging.info("Tuning decision threshold on validation set to maximize F1 (positive class)...")
        val_logits, val_targets = _collect_logits_and_targets(model, val_loader, args.device)
        best_threshold, best_f1 = _best_thresh_for_mcc_from_logits(val_logits, val_targets)
        logging.info(f"Tuned threshold t* = {best_threshold:.3f} with Val F1_pos = {best_f1:.4f}")
    else:
        best_threshold = 0.5
        logging.info("Using default threshold t = 0.5 (no tuning requested).")

    y_true_final, y_pred_final = _evaluate_with_threshold(model, val_loader, args, best_threshold, log_prefix="Final Evaluation on Test Set")
    
    training_metrics["final_threshold"] = float(best_threshold)
    training_metrics["model_config"] = getattr(model, 'config', None)
    
    return model, best_threshold, y_true_final, y_pred_final, training_metrics


class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss

        return focal_loss.mean()

PARAM_GRID = {
    "base_channels": [16],
    "num_blocks": [5],
    "kernel_size": [7],
    "dilation": [2],
    "use_batchnorm": [True],
    "pool_every": [2],
    "activation": ["relu"],
    "pooling": ["max"],
    "double_every": [2]
}

DILATED_GRID = {
    "enc1_channels": [8, 16],
    "enc1_kernel": [3, 5],
    "enc2_channels": [16, 32],
    "enc2_kernel": [3, 5],
    "dec1_channels": [16, 32],
    "dec1_kernel": [3, 5],
    "dec2_channels": [8, 16],
    "dec2_kernel": [3, 5],
    "bottleneck_channels": [32],
    "num_dilated_layers": [5],
    "dilated_kernel": [3],
    "base_dilation": [2],
    "activation": ["silu"],
    "pooling": ["avg"],
}

ENABLE_CNN = False
ENABLE_DILATED = True

def grid_generator():
    if ENABLE_CNN:
        keys, values = zip(*PARAM_GRID.items())
        for combo in product(*values):
            yield (build_cnn, CNNConfig(**dict(zip(keys, combo))))

    if ENABLE_DILATED:
        d_keys, d_vals = zip(*DILATED_GRID.items())
        for combo in product(*d_vals):
            yield (build_dilated_cnn, DilatedCNNConfig(**dict(zip(d_keys, combo))))

def is_dominated(p_metrics: tuple, q_metrics: tuple) -> bool:
    all_ge = True
    any_gt = False
    for pv, qv in zip(p_metrics, q_metrics):
        if qv < pv:
            all_ge = False
            break
        if qv > pv:
            any_gt = True
    return all_ge and any_gt

def find_pareto_front(points: list) -> list:
    front = []
    for p in points:
        p_metrics = p[:3]
        is_p_dominated = any(is_dominated(p_metrics, q[:3]) for q in points if p is not q)
        if not is_p_dominated:
            front.append(p)
    return front

def sweep_configurations(train_loader, val_loader, criterion, args):
    trials = []
    for builder, cfg in grid_generator():
        model = builder(cfg).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        best_state = None
        best_metrics = None
        best_loss = float("inf")
        best_threshold = 0.5
        best_event_f1 = -1.0
        best_evt = None

        for epoch in range(args.epochs_per_trial):
            train_epoch(model, train_loader, criterion, optimizer, args.device)

            base_metrics = evaluate(model, val_loader, criterion, args.device)
            val_logits, val_targets = _collect_logits_and_targets(model, val_loader, args.device)
            tuned_t, _ = _threshold_from_args_or_tune(args, val_logits, val_targets)
            y_true_list, y_pred_list = _evaluate_with_threshold(
                model, val_loader, args, tuned_t, log_prefix="Sweep Evaluation", print_reports=False
            )
            thresh_metrics = _compute_metrics_from_binary_lists(y_true_list, y_pred_list)

            if args.event_tol is not None:
                evt = event_prf1(y_true_list, y_pred_list, tol=int(args.event_tol))
                epoch_score = evt["f1"]
            else:
                epoch_score = thresh_metrics["f1_macro"]

            if epoch_score > best_event_f1 or (abs(epoch_score - best_event_f1) < 1e-9 and base_metrics["loss"] < best_loss):
                best_event_f1 = float(epoch_score)
                best_loss = float(base_metrics["loss"])
                best_state = deepcopy(model.state_dict())
                best_metrics = thresh_metrics
                best_threshold = float(tuned_t)
                best_evt = evt if args.event_tol is not None else None

        n_params = sum(p.numel() for p in model.parameters())
        size_kb = n_params * 4 / 1024

        if args.max_model_size and size_kb > args.max_model_size:
            logging.debug(f"[Sweep] Skipping config (size {size_kb:.0f}kB > {args.max_model_size}kB): {cfg}")
            continue

        config_dict = _config_to_dict(cfg)
        
        trial_metrics = {
            "config": config_dict,
            "final_metrics": {
                "f1_0": best_metrics["f1_class_0"],
                "f1_1": best_metrics["f1_class_1"],
                "f1_macro": best_metrics["f1_macro"],
                "mcc": best_metrics["mcc"],
                "loss": best_loss,
                "threshold": best_threshold,
                "score": best_event_f1,
                "event_metrics": best_evt,
            },
            "model_info": {
                "size_kb": size_kb,
                "num_parameters": n_params,
            },
        }
        
        trial_metrics["_state_dict"] = best_state
        
        trials.append(trial_metrics)

        if best_evt is not None:
            logging.info(
                f"[Sweep] cfg={cfg} | "
                f"F1₀={best_metrics['f1_class_0']:.4f} | F1₁={best_metrics['f1_class_1']:.4f} "
                f"| F1_avg={best_metrics['f1_macro']:.4f} | MCC={best_metrics['mcc']:.4f} | loss={best_loss:.4f} | size={size_kb:.0f} kB | t={best_threshold:.3f} | "
                f"EvtF1={best_evt['f1']:.4f} P={best_evt['precision']:.4f} R={best_evt['recall']:.4f} [TP={best_evt['tp']} FP={best_evt['fp']} FN={best_evt['fn']}]"
            )
        else:
            logging.info(
                f"[Sweep] cfg={cfg} | "
                f"F1₀={best_metrics['f1_class_0']:.4f} | F1₁={best_metrics['f1_class_1']:.4f} "
                f"| F1_avg={best_metrics['f1_macro']:.4f} | MCC={best_metrics['mcc']:.4f} | loss={best_loss:.4f} | size={size_kb:.0f} kB | t={best_threshold:.3f}"
            )
    return trials

def run_grid_search(train_loader, val_loader, criterion, args):
    """
    Performs a full grid search and identifies the best model based on macro F1-score.
    """
    logging.info("Starting Grid Search...")

    trials = sweep_configurations(train_loader, val_loader, criterion, args)

    if not trials:
        logging.error("No trials were completed. Exiting grid search.")
        return None, None

    if args.event_tol is not None:
        best_trial = sorted(trials, key=lambda t: (-t["final_metrics"]["score"], t["final_metrics"]["loss"]))[0]
        evt = best_trial["final_metrics"].get("event_metrics")
        if evt is not None:
            logging.info(
                f"Best (event-level): F1={evt['f1']:.4f} P={evt['precision']:.4f} R={evt['recall']:.4f} "
                f"[TP={evt['tp']} FP={evt['fp']} FN={evt['fn']}]"
            )
    else:
        best_trial = sorted(trials, key=lambda t: (-t["final_metrics"]["f1_macro"], t["final_metrics"]["loss"]))[0]

    best_cfg = best_trial["config"]
    best_state = best_trial["_state_dict"]

    logging.info("\n--- Best Configuration ---")
    logging.info(f"Config: {best_cfg}")
    logging.info(f"F1 Macro: {best_trial['final_metrics']['f1_macro']:.4f} | MCC: {best_trial['final_metrics']['mcc']:.4f} | Size: {best_trial['model_info']['size_kb']:.1f}kB | Val Loss: {best_trial['final_metrics']['loss']:.4f}")

    if isinstance(best_cfg, CNNConfig):
        best_model = build_cnn(best_cfg).to(args.device)
    else:
        best_model = build_dilated_cnn(best_cfg).to(args.device)
    best_model.load_state_dict(best_state)

    return best_cfg, best_model, trials

def run_pareto_search(train_loader, val_loader, criterion, args):
    logging.info("Starting Pareto Front Search...")

    trials = sweep_configurations(train_loader, val_loader, criterion, args)

    if not trials:
        print("No trials were completed. Exiting Pareto search.")
        return []

    points_to_check = [
        (t["final_metrics"]["f1_0"], t["final_metrics"]["f1_1"], -t["model_info"]["size_kb"], 
         t["config"], t["_state_dict"], t["model_info"]["size_kb"], 
         t["final_metrics"]["event_metrics"], t["final_metrics"]["mcc"]) for t in trials
    ]

    front = find_pareto_front(points_to_check)
    front.sort(key=lambda p: (-(p[0] + p[1]), p[5]))

    logging.info("\n--- Pareto-Optimal Configurations ---")
    save_records = []
    for i, (f1_0, f1_1, _neg_size, cfg, state, size_kb, event_metrics, mcc) in enumerate(front):
        if event_metrics is not None:
            event_str = f" | EvtF1={event_metrics['f1']:.4f} P={event_metrics['precision']:.4f} R={event_metrics['recall']:.4f}"
        else:
            event_str = ""
        
        logging.info(f"Model {i:02d}: F1₀={f1_0:.4f} | F1₁={f1_1:.4f} | MCC={mcc:.4f} | Size={size_kb:.1f}kB{event_str} | Config: {cfg}")
        if args.save_top_models:
            save_model_bundle(i, cfg, state, f1_0, f1_1, size_kb, args.top_model_dir, save_records, event_metrics, mcc)

    if args.save_top_models and save_records:
        summary_path = args.top_model_dir / "summary.csv"
        pd.DataFrame(save_records).to_csv(summary_path, index=False)
        logging.info(f"\n[Saved] Models and summary written to {args.top_model_dir}")

    return front, trials




def event_prf1(y_true_list, y_pred_list, tol: int = 2):

    if len(y_true_list) != len(y_pred_list):
        raise ValueError("y_true_list and y_pred_list must have the same length.")
    if tol < 0:
        raise ValueError("tol must be >= 0.")

    tp = fp = fn = 0

    for y_t, y_p in zip(y_true_list, y_pred_list):
        y_t = np.asarray(y_t)
        y_p = np.asarray(y_p)

        if y_t.shape != y_p.shape:
            raise ValueError(f"Shape mismatch: {y_t.shape} vs {y_p.shape}")
        if y_t.ndim != 3 or y_t.shape[1] != 1:
            raise ValueError(f"Expected arrays shaped (N, 1, L); got {y_t.shape}")

        y_t = (y_t > 0).astype(np.uint8)
        y_p = (y_p > 0).astype(np.uint8)

        N, _, L = y_t.shape

        for n in range(N):
            t_idx = np.flatnonzero(y_t[n, 0, :])
            p_idx = np.flatnonzero(y_p[n, 0, :])

            if len(t_idx) == 0 and len(p_idx) == 0:
                continue
            if len(t_idx) == 0:
                fp += len(p_idx)
                continue
            if len(p_idx) == 0:
                fn += len(t_idx)
                continue

            cost = np.abs(t_idx[:, None] - p_idx[None, :])
            cost[cost > tol] = 1e9
            row_ind, col_ind = linear_sum_assignment(cost)
            matches = sum(cost[r, c] <= tol for r, c in zip(row_ind, col_ind))

            tp += matches
            fp += len(p_idx) - matches
            fn += len(t_idx) - matches

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

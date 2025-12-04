import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Optional


def save_model_bundle(
        index: int,
        cfg: 'CNNConfig',
        state_dict: dict,
        f1_0: float,
        f1_1: float,
        size_kb: float,
        save_dir: Path,
        summary_records: list[dict],
        event_metrics: dict = None,
        mcc: float = None
):
    """
    Saves a model's state dict, its configuration, and appends its metadata
    to a list for later creation of a summary CSV.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"model_{index:02d}"

    model_path = save_dir / f"{base_name}.pt"
    config_path = save_dir / f"{base_name}_config.json"

    # Save model weights
    torch.save(state_dict, model_path)

    with open(config_path, "w") as f:
        if isinstance(cfg, dict):
            cfg_dict = cfg
        elif hasattr(cfg, '__dataclass_fields__'):
            cfg_dict = asdict(cfg)
        else:
            cfg_dict = dict(cfg) if hasattr(cfg, '__iter__') else cfg.__dict__
        
        json.dump(cfg_dict, f, indent=2)

    record = {
        "model_file": model_path.name,
        "config_file": config_path.name,
        "F1_class_0": round(f1_0, 4),
        "F1_class_1": round(f1_1, 4),
        "MCC": round(mcc, 4) if mcc is not None else None,
        "size_kB": round(size_kb, 2)
    }
    
    if event_metrics is not None:
        record.update({
            "Event_F1": round(event_metrics["f1"], 4),
            "Event_Precision": round(event_metrics["precision"], 4),
            "Event_Recall": round(event_metrics["recall"], 4),
            "Event_TP": event_metrics["tp"],
            "Event_FP": event_metrics["fp"],
            "Event_FN": event_metrics["fn"]
        })
    
    summary_records.append(record)
    print(f"[Saved] Model bundle '{base_name}' to {save_dir}")


def _ensure_b1l(a: np.ndarray) -> np.ndarray:
    """
    Ensure array shape is (N, 1, L). Accepts (N, L) or (N, 1, L).
    """
    if a.ndim == 3:
        assert a.shape[1] == 1, f"Expected shape (N,1,L), got {a.shape}"
        return a
    elif a.ndim == 2:
        return a[:, None, :]
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {a.shape}")

def _stack_window_lists(arr_list: List[np.ndarray]) -> np.ndarray:
    """
    Concatenate a list of arrays that are each (N,1,L) or (N,L) into a single (N_total,1,L).
    """
    if arr_list is None:
        return None
    normed = [_ensure_b1l(a) for a in arr_list]
    return np.concatenate(normed, axis=0)

@torch.inference_mode()
def plot_test_examples(
    dataset,
    device,
    num_examples: int = 5,
    *,
    y_true_list: Optional[List[np.ndarray]] = None,
    y_pred_list: Optional[List[np.ndarray]] = None,
    model: Optional[torch.nn.Module] = None,
    rng_seed: Optional[int] = None,
    title_prefix: str = "Test Window"
):
    y_true_all = _stack_window_lists(y_true_list) if y_true_list is not None else None
    y_pred_all = _stack_window_lists(y_pred_list) if y_pred_list is not None else None

    if (y_true_all is not None) and (y_pred_all is not None):
        assert y_true_all.shape == y_pred_all.shape, \
            f"Shape mismatch: y_true {y_true_all.shape} vs y_pred {y_pred_all.shape}"

    n_dataset = len(dataset)
    n_pool = n_dataset

    if (y_true_all is not None) or (y_pred_all is not None):
        n_pool = y_true_all.shape[0] if y_true_all is not None else y_pred_all.shape[0]

    rng = np.random.default_rng(rng_seed)
    k = min(num_examples, n_pool)
    if k <= 0:
        print("Nothing to plot: empty dataset or predictions.")
        return
    sel_idx = rng.choice(n_pool, size=k, replace=False)

    fig, axes = plt.subplots(k, 1, figsize=(12, 2.8 * k), sharex=False)
    axes = np.atleast_1d(axes)

    def _get_masks(i: int, x_len: int):
        if y_true_all is not None:
            yt = y_true_all[i, 0, :]
        else:
            yt = None
        if y_pred_all is not None:
            yp = y_pred_all[i, 0, :]
        else:
            yp = None
        if (yp is None) and (model is not None):
            model.eval()
            x_i, _ = dataset[i]
            logits = model(x_i.unsqueeze(0).to(device)).cpu().squeeze()
            prob = torch.sigmoid(logits)
            yp = (prob >= 0.5).int().numpy()
        if yt is None:
            _, y_i = dataset[i]
            yt = y_i.squeeze().numpy()
        if len(yt) != x_len:
            yt = yt[:x_len]
        if len(yp) != x_len:
            yp = yp[:x_len]
        return yt, yp

    for row, (ax, i) in enumerate(zip(axes, sel_idx)):
        x, _ = dataset[i]
        x_flat = x.squeeze().numpy()
        L = x_flat.shape[-1]
        t = np.arange(L)

        yt, yp = _get_masks(int(i), L)
        true_peaks = np.where(yt == 1)[0]
        pred_peaks = np.where(yp == 1)[0]

        ax.plot(t, x_flat, label="PPG Signal", linewidth=1)
        if true_peaks.size > 0:
            ax.scatter(t[true_peaks], x_flat[true_peaks] + 0.03, marker="^", label="True Peak", s=80)
        if pred_peaks.size > 0:
            ax.scatter(t[pred_peaks], x_flat[pred_peaks] - 0.03, marker="x", label="Predicted Peak", s=80)

        ax.set_title(f"{title_prefix} {int(i)}")
        ax.set_ylabel("Normalized Amplitude")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Samples")
    fig.tight_layout()
    plt.show()


@torch.inference_mode()
def plot_nms_comparison(
    dataset,
    model,
    device,
    threshold: float = 0.5,
    nms_window: int = 5,
    num_examples: int = 5,
    rng_seed: Optional[int] = None,
    focus_nms_removed: bool = False
):
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from ppg_pipeline.training import apply_nms_over_windows, postprocess_preds
    
    model.eval()
    
    n_dataset = len(dataset)
    k = min(num_examples, n_dataset)
    if k <= 0:
        print("Nothing to plot: empty dataset.")
        return
    
    if focus_nms_removed and nms_window > 0:
        sel_idx = _find_windows_with_most_nms_removals(
            dataset, model, device, threshold, nms_window, k, rng_seed
        )
    else:
        rng = np.random.default_rng(rng_seed)
        sel_idx = rng.choice(n_dataset, size=k, replace=False)
    
    fig, axes = plt.subplots(k, 2, figsize=(16, 2.8 * k), sharex=False, sharey=False)
    axes = np.atleast_2d(axes)
    
    for row, i in enumerate(sel_idx):
        x, y_true = dataset[i]
        x_flat = x.squeeze().numpy()
        L = x_flat.shape[-1]
        t = np.arange(L)
        
        with torch.no_grad():
            logits = model(x.unsqueeze(0).to(device)).cpu().squeeze()
            probs = torch.sigmoid(logits).numpy()
        
        probs_reshaped = probs.reshape(1, -1) if probs.ndim == 1 else probs
        
        if nms_window > 0:
            preds_with_nms = apply_nms_over_windows(probs_reshaped, threshold, nms_window)
            preds_with_nms = preds_with_nms.flatten()
        else:
            preds_with_nms = (probs >= threshold).astype(np.uint8)
        
        preds_without_nms = (probs >= threshold).astype(np.uint8)
        
        yt = y_true.squeeze().numpy()
        if len(yt) != L:
            yt = yt[:L]
        
        true_peaks = np.where(yt == 1)[0]
        pred_peaks_with_nms = np.where(preds_with_nms == 1)[0]
        pred_peaks_without_nms = np.where(preds_without_nms == 1)[0]
        
        tp_without = np.intersect1d(pred_peaks_without_nms, true_peaks)
        fp_without = np.setdiff1d(pred_peaks_without_nms, true_peaks)
        fn_without = np.setdiff1d(true_peaks, pred_peaks_without_nms)
        
        tp_with = np.intersect1d(pred_peaks_with_nms, true_peaks)
        fp_with = np.setdiff1d(pred_peaks_with_nms, true_peaks)
        fn_with = np.setdiff1d(true_peaks, pred_peaks_with_nms)
        
        ax_left = axes[row, 0]
        ax_left.plot(t, x_flat, label="PPG Signal", linewidth=1, color='gray')
        
        if true_peaks.size > 0:
            ax_left.scatter(t[true_peaks], x_flat[true_peaks] + 0.03, 
                          marker="^", label="True Peak", s=80, color='#FF69B4', alpha=0.7)
        
        if tp_without.size > 0:
            ax_left.scatter(t[tp_without], x_flat[tp_without] - 0.03, 
                          marker="x", label="TP (correct)", s=80, color='#00AA00', linewidths=2.5)
        
        if fp_without.size > 0:
            ax_left.scatter(t[fp_without], x_flat[fp_without] - 0.04, 
                          marker="x", label="FP (wrong)", s=80, color='#CC0000', linewidths=2.5)
        
        ax_left.set_title(f"Without NMS - Window {int(i)}")
        ax_left.set_ylabel("Normalized Amplitude")
        ax_left.grid(True, linestyle="--", alpha=0.6)
        ax_left.legend(loc="upper right", fontsize=7)
        
        ax_right = axes[row, 1]
        ax_right.plot(t, x_flat, label="PPG Signal", linewidth=1, color='gray')
        
        if true_peaks.size > 0:
            ax_right.scatter(t[true_peaks], x_flat[true_peaks] + 0.03, 
                           marker="^", label="True Peak", s=80, color='#FF69B4', alpha=0.7)
        
        if tp_with.size > 0:
            ax_right.scatter(t[tp_with], x_flat[tp_with] - 0.03, 
                           marker="x", label="TP (correct)", s=80, color='#00AA00', linewidths=2.5)
        
        if fp_with.size > 0:
            ax_right.scatter(t[fp_with], x_flat[fp_with] - 0.04, 
                           marker="x", label="FP (wrong)", s=80, color='#CC0000', linewidths=2.5)
        
        ax_right.set_title(f"With NMS (window={nms_window}) - Window {int(i)}")
        ax_right.set_ylabel("Normalized Amplitude")
        ax_right.grid(True, linestyle="--", alpha=0.6)
        ax_right.legend(loc="upper right", fontsize=7)
    
    axes[-1, 0].set_xlabel("Samples")
    axes[-1, 1].set_xlabel("Samples")
    fig.tight_layout()
    plt.show()


def _find_windows_with_most_nms_removals(
    dataset,
    model,
    device,
    threshold: float,
    nms_window: int,
    num_examples: int,
    rng_seed: Optional[int] = None
):
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from ppg_pipeline.training import apply_nms_over_windows
    
    model.eval()
    n_dataset = len(dataset)
    removals = []
    
    with torch.no_grad():
        for i in range(min(n_dataset, 1000)):  # Limit search for efficiency
            x, _ = dataset[i]
            logits = model(x.unsqueeze(0).to(device)).cpu().squeeze()
            probs = torch.sigmoid(logits).numpy()
            
            # Get predictions with and without NMS
            probs_reshaped = probs.reshape(1, -1) if probs.ndim == 1 else probs
            preds_without_nms = (probs >= threshold).astype(np.uint8)
            preds_with_nms = apply_nms_over_windows(probs_reshaped, threshold, nms_window).flatten()
            
            # Count removed points
            num_without_nms = preds_without_nms.sum()
            num_with_nms = preds_with_nms.sum()
            removals.append((i, num_without_nms - num_with_nms))
    
    # Sort by removals (descending) and return top k indices
    removals.sort(key=lambda x: x[1], reverse=True)
    return np.array([idx for idx, _ in removals[:num_examples]])
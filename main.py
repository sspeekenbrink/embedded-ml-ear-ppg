import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch import nn
import json
from pathlib import Path
import sys

from ppg_pipeline.config import parse_cli_args
from ppg_pipeline.data import (
    find_data_sources,
    create_dataset_from_sources,
    PPGWindowDataset,
)
from ppg_pipeline.models import PPGPeakDetectorCNN, DilatedCNN
from ppg_pipeline.training import (
    run_single_training,
    run_grid_search,
    run_pareto_search,
    FocalLoss,
)
from ppg_pipeline.utils import plot_test_examples, plot_nms_comparison

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """
    Main pipeline execution function.
    """
    args = parse_cli_args()

    args_dict = vars(args)
    args_for_log = {
        k: str(v) if isinstance(v, Path) else v for k, v in args_dict.items()
    }
    args_str = json.dumps(args_for_log, indent=2)
    logging.info(f"Running with the following configuration:\n{args_str}")

    if args.plot_nms_comparison and args.nms == 0:
        logging.warning(
            "--plot_nms_comparison requires --nms to be set (e.g., --nms 5). Setting --nms to 5."
        )
        args.nms = 5

    logging.info(f"Scanning for data sources in '{args.data_dir}'...")
    sources = find_data_sources(args.data_dir)
    if not sources:
        logging.error(
            "No valid data sources found. Please check the --data_dir path and folder structure."
        )
        sys.exit(1)

    logging.info("Creating combined dataset from all sources...")
    windows, labels = create_dataset_from_sources(sources, args)

    if windows.size == 0:
        logging.error("Failed to create any windows from the provided data. Exiting.")
        sys.exit(1)

    dataset = PPGWindowDataset(windows, labels)
    logging.info(
        f"Successfully created a combined dataset with {len(dataset)} windows of size {windows.shape[1]}."
    )

    n_train = int(len(dataset) * args.train_split)
    n_test = len(dataset) - n_train
    train_ds, test_ds = random_split(
        dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42)
    )
    logging.info(f"Train/Test split: {len(train_ds)} / {len(test_ds)}")

    if args.sample_balance:
        logging.info("Using WeightedRandomSampler to balance training data.")
        peak_mask = train_ds.dataset.has_peak[train_ds.indices]
        weights = torch.where(peak_mask, torch.tensor(1.0), torch.tensor(0.1))
        sampler = WeightedRandomSampler(
            weights, num_samples=len(train_ds), replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    pos_samples = labels.sum()
    total_samples = labels.size
    pos_weight_value = (total_samples - pos_samples) / (pos_samples + 1e-6)
    logging.info(
        f"Positive labels in combined dataset: {pos_samples}/{total_samples} (pos_weight for BCE: {pos_weight_value:.2f})"
    )

    match args.loss:
        case "bce":
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight_value, device=args.device)
            )
        case "focal":
            criterion = FocalLoss(alpha=0.1, gamma=5.0)
        case _:
            raise ValueError(f"Unknown loss function: {args.loss}")

    logging.info(f"Using {args.loss.upper()} loss function.")

    model = None
    y_true = None
    y_pred = None
    training_metrics = None
    tuned_threshold = None

    if args.grid_search:
        if args.pareto_front:
            _, trials = run_pareto_search(train_loader, test_loader, criterion, args)
            training_metrics = {"search_type": "pareto_front", "trials": trials}
        else:
            _, model, trials = run_grid_search(
                train_loader, test_loader, criterion, args
            )
            training_metrics = {"search_type": "grid_search", "trials": trials}
    else:
        model = DilatedCNN().to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        model, tuned_threshold, y_true, y_pred, single_metrics = run_single_training(
            model, train_loader, test_loader, criterion, optimizer, args
        )
        training_metrics = {"search_type": "single_model", "metrics": single_metrics}

    if args.plot_examples and model is not None:
        logging.info("Plotting test examples...")
        plot_test_examples(
            dataset=test_ds,
            device=args.device,
            num_examples=5,
            y_true_list=y_true,
            y_pred_list=y_pred,
            model=model,
            rng_seed=43,
        )

    if args.plot_nms_comparison and model is not None:
        logging.info("Plotting NMS comparison...")
        if tuned_threshold is not None:
            threshold = tuned_threshold
            logging.info(f"Using tuned threshold={threshold:.3f} from training")
        elif args.decision_threshold is not None:
            threshold = args.decision_threshold
            logging.info(f"Using specified threshold={threshold:.3f}")
        else:
            threshold = 0.5
            logging.info(f"Using default threshold={threshold:.3f}")

        focus_mode = (
            "focusing on windows with most NMS removals"
            if args.focus_nms_removed
            else "random windows"
        )
        logging.info(f"NMS window={args.nms}, {focus_mode}")
        plot_nms_comparison(
            dataset=test_ds,
            model=model,
            device=args.device,
            threshold=threshold,
            nms_window=args.nms,
            num_examples=5,
            rng_seed=43,
            focus_nms_removed=args.focus_nms_removed,
        )
        logging.info("NMS comparison plots generated.")

    if training_metrics is not None:

        def clean_metrics_for_json(obj):
            if isinstance(obj, dict):
                cleaned = {}
                for key, value in obj.items():
                    if key.startswith("_state_dict"):
                        continue
                    cleaned[key] = clean_metrics_for_json(value)
                return cleaned
            elif isinstance(obj, list):
                return [clean_metrics_for_json(item) for item in obj]
            else:
                return obj

        metrics_output = {
            "training_config": args_for_log,
            "dataset_info": {
                "total_windows": len(dataset),
                "train_windows": len(train_ds),
                "test_windows": len(test_ds),
                "positive_samples": int(pos_samples),
                "total_samples": int(total_samples),
                "positive_weight": float(pos_weight_value),
            },
            "training_metrics": clean_metrics_for_json(training_metrics),
        }

        try:
            with open(args.save_metrics_json, "w") as f:
                json.dump(metrics_output, f, indent=2, default=str)
            logging.info(f"Training metrics saved to: {args.save_metrics_json}")
        except Exception as e:
            logging.error(
                f"Failed to save training metrics to {args.save_metrics_json}: {e}"
            )

    logging.info("Pipeline finished.")


if __name__ == "__main__":
    main()

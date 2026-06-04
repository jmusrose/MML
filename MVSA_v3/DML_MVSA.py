#!/usr/bin/env python3
"""
DML_v1/MVSA/DML_MVSA.py

Main training script for DML multimodal sentiment analysis on MVSA_Single.

Architecture: BERT (text) + ResNet-152 (image) with decision-level fusion.
Training loss: CE(fused, target) + CE(text, target) + CE(image, target)
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data.helpers import get_data_loaders
from models.dml_classifier import Classifier
from utils.conformal import calibrate_conformal, evaluate_conformal
from utils.logger import create_logger
from utils.loss import information_bottleneck_classification_loss
from utils.utils import (
    Averager,
    append_experiment_record,
    load_checkpoint,
    log_metrics,
    save_checkpoint,
    set_seed,
    store_preds_to_disk,
)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)


def get_args(parser):
    """Register all CLI arguments."""
    parser.add_argument("--batch_sz", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--bert_model",
        type=str,
        default="bert-base-uncased",
        help="Pre-trained BERT model name or path",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(_REPO_ROOT, "datasets_shared"),
        help="Dataset root directory",
    )
    parser.add_argument(
        "--drop_img_percent",
        type=float,
        default=0.0,
        help="Fraction of images to drop",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--freeze_img", type=int, default=3, help="Epochs to freeze image encoder"
    )
    parser.add_argument(
        "--freeze_txt", type=int, default=5, help="Epochs to freeze text encoder"
    )
    parser.add_argument(
        "--hidden_sz", type=int, default=768, help="BERT hidden size"
    )
    parser.add_argument(
        "--img_embed_pool_type",
        type=str,
        default="avg",
        choices=["max", "avg"],
        help="Image pooling type",
    )
    parser.add_argument(
        "--img_hidden_sz",
        type=int,
        default=2048,
        help="ResNet feature dimension",
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument(
        "--ib_beta",
        type=float,
        default=1e-3,
        help="Weight for information bottleneck KL loss",
    )
    parser.add_argument(
        "--ib_eps_scale",
        type=float,
        default=1.0,
        help="Scale for reparameterization noise during training",
    )
    parser.add_argument(
        "--conformal_alpha",
        type=float,
        default=0.1,
        help="Miscoverage level for split conformal prediction",
    )
    parser.add_argument(
        "--calib_size",
        type=int,
        default=0,
        help="Calibration sample count; <=0 uses existing cp/dev/val or 20% train split",
    )
    parser.add_argument(
        "--lr_factor", type=float, default=0.5, help="LR reduction factor"
    )
    parser.add_argument(
        "--lr_patience", type=int, default=2, help="LR scheduler patience"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum training epochs"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=64, help="Maximum text sequence length"
    )
    parser.add_argument(
        "--n_classes", type=int, default=3, help="Number of sentiment classes"
    )
    parser.add_argument("--n_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--name", type=str, default="dml_mvsa", help="Experiment name")
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Experiment note appended to all_experiments.json",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.0,
        help="Noise severity for robustness eval",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="Gaussian",
        help="Noise type (Gaussian/Salt)",
    )
    parser.add_argument(
        "--num_image_embeds",
        type=int,
        default=3,
        help="Number of image embedding patches",
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "savepath"),
        help="Save directory",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--task",
        type=str,
        default="MVSA_Single",
        help="Task/dataset name",
    )


def get_criterion(args):
    """Returns CrossEntropyLoss."""
    return nn.CrossEntropyLoss()


def get_optimizer(model, args):
    """Returns Adam optimizer."""
    return optim.Adam(model.parameters(), lr=args.lr)


def get_scheduler(optimizer, args):
    """Returns ReduceLROnPlateau(mode='max')."""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=args.lr_patience,
        factor=args.lr_factor,
    )


def train_epoch(epoch, train_loader, model, optimizer, criterion, logger, args):
    """Single epoch training with 3-branch CE loss.

    Loss = CE(fused, target) + CE(text, target) + CE(image, target)
    """
    model.train()
    tl = Averager()
    tl_fused = Averager()
    tl_txt = Averager()
    tl_img = Averager()
    tl_ib = Averager()
    correct_fused = 0
    correct_txt = 0
    correct_img = 0
    total_samples = 0

    # Freeze/unfreeze encoders based on epoch
    if epoch < args.freeze_img:
        for param in model.imgclf.img_encoder.parameters():
            param.requires_grad = False
    else:
        for param in model.imgclf.img_encoder.parameters():
            param.requires_grad = True

    if epoch < args.freeze_txt:
        for param in model.txtclf.enc.parameters():
            param.requires_grad = False
    else:
        for param in model.txtclf.enc.parameters():
            param.requires_grad = True

    for step, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch}")):
        text, segment, mask, image, target, indices = batch
        device = next(model.parameters()).device
        text = text.to(device)
        mask = mask.to(device)
        segment = segment.to(device)
        image = image.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        fused_logits, txt_logits, img_logits, _, _, ib_loss = model(
            text, mask, segment, image
        )

        loss, loss_parts = information_bottleneck_classification_loss(
            criterion,
            fused_logits,
            txt_logits,
            img_logits,
            target,
            ib_loss,
            args.ib_beta,
        )

        if torch.isnan(loss).any():
            logger.warning(
                f"NaN detected at step {step}: "
                f"loss_fused={loss_parts['fused'].item()}, "
                f"loss_txt={loss_parts['txt'].item()}, "
                f"loss_img={loss_parts['img'].item()}, "
                f"loss_ib={loss_parts['ib'].item()}"
            )

        loss.backward()
        optimizer.step()

        tl.add(loss.item())
        tl_fused.add(loss_parts['fused'].item())
        tl_txt.add(loss_parts['txt'].item())
        tl_img.add(loss_parts['img'].item())
        tl_ib.add(loss_parts['ib'].item())
        with torch.no_grad():
            correct_fused += (torch.argmax(fused_logits, dim=1) == target).sum().item()
            correct_txt += (torch.argmax(txt_logits, dim=1) == target).sum().item()
            correct_img += (torch.argmax(img_logits, dim=1) == target).sum().item()
            total_samples += target.size(0)

    avg_loss = tl.item()
    loss_fused_ave = tl_fused.item()
    loss_txt_ave = tl_txt.item()
    loss_img_ave = tl_img.item()
    loss_ib_ave = tl_ib.item()
    acc_fused = correct_fused / total_samples if total_samples else 0.0
    acc_txt = correct_txt / total_samples if total_samples else 0.0
    acc_img = correct_img / total_samples if total_samples else 0.0
    logger.info(
        f"Epoch {epoch}: Total Loss: {avg_loss:.4f}, "
        f"loss_fused:{loss_fused_ave:.4f}, "
        f"loss_txt:{loss_txt_ave:.4f}, "
        f"loss_img:{loss_img_ave:.4f}, "
        f"loss_ib:{loss_ib_ave:.4f}, "
        f"acc_fused:{acc_fused:.4f}, "
        f"acc_txt:{acc_txt:.4f}, "
        f"acc_img:{acc_img:.4f}"
    )
    return model


def eval_epoch(epoch, loader, model, criterion, logger, args):
    """Evaluation returning metrics dict with 'acc' and 'loss' keys."""
    model.eval()
    losses = []
    preds = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating epoch {epoch}"):
            text, segment, mask, image, target, indices = batch
            device = next(model.parameters()).device
            text = text.to(device)
            mask = mask.to(device)
            segment = segment.to(device)
            image = image.to(device)
            target = target.to(device)

            fused_logits, _, _, _, _, _ = model(text, mask, segment, image)

            loss = criterion(fused_logits, target)
            losses.append(loss.item())

            pred = fused_logits.argmax(dim=1).cpu().numpy()
            preds.append(pred)
            targets.append(target.cpu().numpy())

    all_preds = np.concatenate(preds)
    all_targets = np.concatenate(targets)

    metrics = {
        "loss": np.mean(losses),
        "acc": accuracy_score(all_targets, all_preds),
    }
    return metrics


def collect_multimodal_logits(loader, model):
    """Collect fused/text/image logits and labels from a loader."""
    model.eval()
    logits = {"fused": [], "text": [], "image": []}
    labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting conformal logits"):
            text, segment, mask, image, target, indices = batch
            device = next(model.parameters()).device
            text = text.to(device)
            mask = mask.to(device)
            segment = segment.to(device)
            image = image.to(device)
            target = target.to(device)

            fused_logits, txt_logits, img_logits, _, _, _ = model(
                text, mask, segment, image
            )
            logits["fused"].append(fused_logits.detach().cpu().numpy())
            logits["text"].append(txt_logits.detach().cpu().numpy())
            logits["image"].append(img_logits.detach().cpu().numpy())
            labels.append(target.detach().cpu().numpy())

    return {
        modality: np.concatenate(values, axis=0)
        for modality, values in logits.items()
    }, np.concatenate(labels, axis=0)


def main():
    """Full pipeline: setup -> train -> save best -> robustness eval -> log results."""
    parser = argparse.ArgumentParser(
        description="DML Multimodal Sentiment Analysis on MVSA"
    )
    get_args(parser)
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    # Data loaders (clean for training)
    args.noise_level = 0.0
    args.noise_type = "Gaussian"
    train_loader, val_loader, cp_loader, test_loaders = get_data_loaders(args)

    # Model
    model = Classifier(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    logger = create_logger(os.path.join(args.savedir, "training.log"), args)
    torch.save(args, os.path.join(args.savedir, "args.pt"))

    # Training loop
    best_metric = -np.inf
    n_no_improve = 0

    logger.info("Starting DML MVSA training...")

    for epoch in range(args.max_epochs):
        model = train_epoch(
            epoch, train_loader, model, optimizer, criterion, logger, args
        )

        val_metrics = eval_epoch(epoch, val_loader, model, criterion, logger, args)
        log_metrics("Validation", val_metrics, args, logger)

        tuning_metric = val_metrics["acc"]
        scheduler.step(tuning_metric)

        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
            logger.info(f"*** NEW BEST MODEL *** Epoch {epoch} | Acc: {best_metric:.4f}")
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "n_no_improve": n_no_improve,
                    "best_metric": best_metric,
                },
                is_improvement,
                args.savedir,
                filename="model_best_clean.pt",
            )
        else:
            n_no_improve += 1

        if n_no_improve >= args.patience:
            logger.info(
                f"No improvement for {args.patience} epochs. Stopping early."
            )
            break

    # Load best model for robustness evaluation
    logger.info("Loading best model for final robust evaluation...")
    model_path = os.path.join(args.savedir, "model_best_clean.pt")
    if os.path.exists(model_path):
        load_checkpoint(model, model_path)
    model.eval()

    calib_logits, calib_labels = collect_multimodal_logits(cp_loader, model)
    conformal_calibration = calibrate_conformal(
        calib_logits,
        calib_labels,
        args.conformal_alpha,
    )
    conformal_results = {
        "alpha": float(args.conformal_alpha),
        "calibration": {
            "source": getattr(args, "calibration_source", "unknown"),
            "size": int(conformal_calibration["n_calibration"]),
        },
        "thresholds": conformal_calibration["thresholds"],
        "robustness": {},
    }

    # Robustness evaluation scenarios
    test_scenarios = [
        {"name": "Clean Test", "noise_level": 0.0, "noise_type": "Gaussian"},
        {"name": "Gaussian (Lvl 5.0)", "noise_level": 5.0, "noise_type": "Gaussian"},
        {"name": "Gaussian (Lvl 10.0)", "noise_level": 10.0, "noise_type": "Gaussian"},
        {"name": "Salt & Pepper (Lvl 5.0)", "noise_level": 5.0, "noise_type": "Salt"},
        {"name": "Salt & Pepper (Lvl 10.0)", "noise_level": 10.0, "noise_type": "Salt"},
    ]

    final_results = {}

    for scenario in test_scenarios:
        args.noise_level = scenario["noise_level"]
        args.noise_type = scenario["noise_type"]

        logger.info(f"--- Evaluating: {scenario['name']} ---")
        _, _, _, current_test_loaders = get_data_loaders(args)
        current_test_loader = current_test_loaders["test"]

        scenario_metrics = eval_epoch(
            -1, current_test_loader, model, criterion, logger, args
        )
        metric_val = scenario_metrics["acc"]
        final_results[scenario["name"]] = metric_val

        log_metrics(scenario["name"], scenario_metrics, args, logger)
        scenario_logits, scenario_labels = collect_multimodal_logits(
            current_test_loader,
            model,
        )
        conformal_results["robustness"][scenario["name"]] = evaluate_conformal(
            scenario_logits,
            scenario_labels,
            conformal_calibration["thresholds"],
        )

    # Log results table
    logger.info("\n" + "=" * 60)
    logger.info(f"{'FINAL ROBUSTNESS EVALUATION RESULTS ON TEST SET':^60}")
    logger.info("=" * 60)
    logger.info(f"| {'Test Scenario':<35} | {'Score (Acc)':<18} |")
    logger.info("|" + "-" * 37 + "|" + "-" * 20 + "|")

    for name, score in final_results.items():
        logger.info(f"| {name:<35} | {score:>18.4f} |")

    logger.info("=" * 60 + "\n")

    # Save results JSON
    result_file = os.path.join(args.savedir, "final_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_clean_model": {
                    "clean_acc": float(final_results["Clean Test"])
                },
                "robustness": final_results,
                "conformal": conformal_results,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )
    logger.info(f"Saved final results to {result_file}")

    summary_path = os.path.join(os.path.dirname(args.savedir), "all_experiments.json")
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": "mvsa",
        "name": args.name,
        "note": args.note,
        "seed": args.seed,
        "lr": args.lr,
        "ib_beta": args.ib_beta,
        "ib_eps_scale": args.ib_eps_scale,
        "batch_sz": args.batch_sz,
        "max_epochs": args.max_epochs,
        "best_clean_acc": float(final_results["Clean Test"]),
        "robustness": final_results,
        "conformal": conformal_results,
        "savedir": args.savedir,
    }
    append_experiment_record(summary_path, record)
    logger.info(f"Appended experiment record to {summary_path}")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()

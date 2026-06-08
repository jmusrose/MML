#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DML_CREMAD/DML_cremad.py

CREMA-D Logit Fusion 主训练脚本。

决策级融合：0.5 * audio_logits + 0.5 * video_logits
三路 CE 损失：CE(fused, target) + CE(audio, target) + CE(video, target)

代码风格遵循 CPSC_CREMAD/CPSC_cremad.py。
"""

import os
import sys

# Allow running this script from any working directory by anchoring imports
# and default paths to the script's own location.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch.nn import functional as F
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
import random
from datetime import datetime
from sklearn.metrics import f1_score, average_precision_score

from data.template import config
from dataset.CREMA import CramedDataset
from dataset.CREMA_noised import CramedDatasetNoised
from model.DMLClassifier import DMLClassifier
from utils.utils import (
    create_logger,
    Averager,
    append_experiment_record,
    deep_update_dict,
)
from utils.loss import information_bottleneck_classification_loss
from utils.tools import weight_init, compute_mAP, setup_seed
from utils.conformal import (
    calibrate_conformal,
    conformal_uncertainty_from_logits,
    evaluate_conformal,
)


def _batch_conformal_ib_betas(audio_logits, video_logits, conformal_thresholds, cfg):
    if conformal_thresholds is None:
        return None, {}

    ib_beta = cfg.get("ib_beta", 1e-3)
    uncertainties = conformal_uncertainty_from_logits(
        {
            "audio": audio_logits.detach().cpu().numpy(),
            "video": video_logits.detach().cpu().numpy(),
        },
        conformal_thresholds,
    )
    ib_betas = {
        "audio": ib_beta * (1.0 + uncertainties["audio"]),
        "video": ib_beta * (1.0 + uncertainties["video"]),
    }
    return ib_betas, uncertainties


def train_audio_video(epoch, train_loader, model, optimizer, logger, cfg, conformal_thresholds=None):
    """单 epoch 训练循环（决策级融合 + 三路 CE 损失之和）。

    损失 = CE(fused, target) + CE(audio, target) + CE(video, target)
    """
    model.train()
    tl = Averager()
    tl_fused = Averager()
    tl_audio = Averager()
    tl_video = Averager()
    tl_ib = Averager()
    tl_ib_audio = Averager()
    tl_ib_video = Averager()
    tl_beta_audio = Averager()
    tl_beta_video = Averager()
    tl_uncertainty_audio = Averager()
    tl_uncertainty_video = Averager()
    correct_fused = 0
    correct_audio = 0
    correct_video = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss().cuda()
    ib_beta = cfg.get("ib_beta", 1e-3)

    for step, (spectrogram, image, y) in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch}")):
        image = image.float().cuda()
        y = y.cuda()
        spectrogram = spectrogram.unsqueeze(1).float().cuda()

        optimizer.zero_grad()

        fused_logits, audio_logits, video_logits, _, _, ib_losses = model(spectrogram, image)
        ib_betas, uncertainties = _batch_conformal_ib_betas(
            audio_logits,
            video_logits,
            conformal_thresholds,
            cfg,
        )
        loss, loss_parts = information_bottleneck_classification_loss(
            criterion,
            fused_logits,
            audio_logits,
            video_logits,
            y,
            ib_losses,
            ib_beta,
            ib_betas=ib_betas,
        )
        if uncertainties:
            loss_parts["uncertainty_audio"] = uncertainties["audio"]
            loss_parts["uncertainty_video"] = uncertainties["video"]

        # NaN detection
        if torch.isnan(loss).any():
            logger.info(
                f"NaN detected at step {step}: "
                f"loss_fused={loss_parts['fused'].item()}, "
                f"loss_audio={loss_parts['audio'].item()}, "
                f"loss_video={loss_parts['video'].item()}, "
                f"loss_ib={loss_parts['ib'].item()}"
            )

        loss.backward()
        optimizer.step()

        tl.add(loss.item())
        tl_fused.add(loss_parts['fused'].item())
        tl_audio.add(loss_parts['audio'].item())
        tl_video.add(loss_parts['video'].item())
        tl_ib.add(loss_parts['ib'].item())
        if "ib_audio" in loss_parts:
            tl_ib_audio.add(loss_parts["ib_audio"].item())
            tl_ib_video.add(loss_parts["ib_video"].item())
            tl_beta_audio.add(loss_parts["beta_audio"])
            tl_beta_video.add(loss_parts["beta_video"])
            tl_uncertainty_audio.add(loss_parts["uncertainty_audio"])
            tl_uncertainty_video.add(loss_parts["uncertainty_video"])
        with torch.no_grad():
            labels = torch.argmax(y, dim=1)
            correct_fused += (torch.argmax(fused_logits, dim=1) == labels).sum().item()
            correct_audio += (torch.argmax(audio_logits, dim=1) == labels).sum().item()
            correct_video += (torch.argmax(video_logits, dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    loss_ave = tl.item()
    loss_fused_ave = tl_fused.item()
    loss_audio_ave = tl_audio.item()
    loss_video_ave = tl_video.item()
    loss_ib_ave = tl_ib.item()
    loss_ib_audio_ave = tl_ib_audio.item()
    loss_ib_video_ave = tl_ib_video.item()
    beta_audio_ave = tl_beta_audio.item()
    beta_video_ave = tl_beta_video.item()
    uncertainty_audio_ave = tl_uncertainty_audio.item()
    uncertainty_video_ave = tl_uncertainty_video.item()
    acc_fused = correct_fused / total_samples if total_samples else 0.0
    acc_audio = correct_audio / total_samples if total_samples else 0.0
    acc_video = correct_video / total_samples if total_samples else 0.0
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(
        f'Epoch {epoch}: Average Training Loss: {loss_ave:.4f}, '
        f'loss_fused:{loss_fused_ave:.4f}, '
        f'loss_audio:{loss_audio_ave:.4f}, '
        f'loss_video:{loss_video_ave:.4f}, '
        f'loss_ib:{loss_ib_ave:.4f}, '
        f'loss_ib_audio:{loss_ib_audio_ave:.4f}, '
        f'loss_ib_video:{loss_ib_video_ave:.4f}, '
        f'beta_audio:{beta_audio_ave:.6f}, '
        f'beta_video:{beta_video_ave:.6f}, '
        f'uncertainty_audio:{uncertainty_audio_ave:.4f}, '
        f'uncertainty_video:{uncertainty_video_ave:.4f}, '
        f'acc_fused:{acc_fused:.4f}, '
        f'acc_audio:{acc_audio:.4f}, '
        f'acc_video:{acc_video:.4f}'
    )
    return model


best_acc = 0.0
best_epoch = 0
best_log = ""
best_metrics = {}


def val(epoch, val_loader, model, logger):
    """验证循环：计算 accuracy、F1-score、mAP（fused/audio/video 三路）。"""
    global best_acc, best_epoch, best_log, best_metrics
    model.eval()
    pred_list = []
    pred_list_a = []
    pred_list_v = []
    label_list = []
    soft_pred = []
    soft_pred_a = []
    soft_pred_v = []
    one_hot_label = []

    with torch.no_grad():
        for step, (spectrogram, image, y) in enumerate(tqdm(val_loader, desc=f"Validation epoch {epoch}")):
            label_list = label_list + torch.argmax(y, dim=1).tolist()
            one_hot_label = one_hot_label + y.tolist()
            image = image.float().cuda()
            y = y.cuda()
            spectrogram = spectrogram.unsqueeze(1).float().cuda()

            fused_logits, audio_logits, video_logits, _, _, _ = model(spectrogram, image)

            soft_pred_a = soft_pred_a + (F.softmax(audio_logits, dim=1)).tolist()
            soft_pred_v = soft_pred_v + (F.softmax(video_logits, dim=1)).tolist()
            soft_pred = soft_pred + (F.softmax(fused_logits, dim=1)).tolist()

            pred = (F.softmax(fused_logits, dim=1)).argmax(dim=1)
            pred_a = (F.softmax(audio_logits, dim=1)).argmax(dim=1)
            pred_v = (F.softmax(video_logits, dim=1)).argmax(dim=1)

            pred_list = pred_list + pred.tolist()
            pred_list_a = pred_list_a + pred_a.tolist()
            pred_list_v = pred_list_v + pred_v.tolist()

    f1 = f1_score(label_list, pred_list, average='macro')
    f1_a = f1_score(label_list, pred_list_a, average='macro')
    f1_v = f1_score(label_list, pred_list_v, average='macro')

    correct = sum(1 for x, y in zip(label_list, pred_list) if x == y)
    correct_a = sum(1 for x, y in zip(label_list, pred_list_a) if x == y)
    correct_v = sum(1 for x, y in zip(label_list, pred_list_v) if x == y)

    acc = correct / len(label_list)
    acc_a = correct_a / len(label_list)
    acc_v = correct_v / len(label_list)

    mAP = compute_mAP(torch.Tensor(soft_pred), torch.Tensor(one_hot_label))
    mAP_a = compute_mAP(torch.Tensor(soft_pred_a), torch.Tensor(one_hot_label))
    mAP_v = compute_mAP(torch.Tensor(soft_pred_v), torch.Tensor(one_hot_label))

    log_message = (
        f'Epoch {epoch}: f1:{f1:.4f}, acc:{acc:.4f}, mAP:{mAP:.4f}, '
        f'f1_a:{f1_a:.4f}, acc_a:{acc_a:.4f}, mAP_a:{mAP_a:.4f}, '
        f'f1_v:{f1_v:.4f}, acc_v:{acc_v:.4f}, mAP_v:{mAP_v:.4f}'
    )

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(log_message)

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        best_log = log_message
        best_metrics = {
            'f1': f1, 'acc': acc, 'mAP': mAP,
            'f1_a': f1_a, 'acc_a': acc_a, 'mAP_a': mAP_a,
            'f1_v': f1_v, 'acc_v': acc_v, 'mAP_v': mAP_v
        }
        logger.info(f'NEW BEST ACC | Epoch {epoch} | Acc: {acc:.4f}')

    return acc, acc_a, acc_v


def resolve_train_split_calib_size(num_samples, requested_size):
    if num_samples < 2:
        raise ValueError("train split needs at least 2 samples for train/calibration")
    if requested_size > 0:
        return min(int(requested_size), num_samples - 1)
    return min(max(1, int(round(num_samples * 0.2))), num_samples - 1)


def collect_multimodal_logits(loader, model):
    """Collect fused/audio/video logits and integer labels from a loader."""
    model.eval()
    logits = {"fused": [], "audio": [], "video": []}
    labels = []

    with torch.no_grad():
        for spectrogram, image, y in tqdm(loader, desc="Collecting conformal logits"):
            labels.extend(torch.argmax(y, dim=1).tolist())
            image = image.float().cuda()
            spectrogram = spectrogram.unsqueeze(1).float().cuda()

            fused_logits, audio_logits, video_logits, _, _, _ = model(
                spectrogram,
                image,
            )
            logits["fused"].append(fused_logits.detach().cpu().numpy())
            logits["audio"].append(audio_logits.detach().cpu().numpy())
            logits["video"].append(video_logits.detach().cpu().numpy())

    return {
        modality: np.concatenate(values, axis=0)
        for modality, values in logits.items()
    }, np.asarray(labels, dtype=np.int64)



if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join(SCRIPT_DIR, 'data', 'crema.json'),
    )
    parser.add_argument('--conformal_alpha', type=float, default=None)
    parser.add_argument('--uncertainty_tau', type=float, default=None)
    parser.add_argument('--calib_size', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    args = parser.parse_args()

    cfg = config

    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)
    if args.conformal_alpha is not None:
        cfg['conformal_alpha'] = args.conformal_alpha
    cfg.setdefault('conformal_alpha', 0.1)
    if args.uncertainty_tau is not None:
        cfg['uncertainty_tau'] = args.uncertainty_tau
    cfg.setdefault('uncertainty_tau', 1.0)
    if args.calib_size is not None:
        cfg['calib_size'] = args.calib_size
    cfg.setdefault('calib_size', 0)
    if args.patience is not None:
        cfg.setdefault('train', {})['early_stopping_patience'] = args.patience
    cfg.setdefault('train', {}).setdefault('early_stopping_patience', 15)

    # Anchor output_dir to the script directory if user left it as '.'
    # so logs/checkpoints land next to this script regardless of cwd.
    if cfg.get('output_dir', '.') in ('.', './', ''):
        cfg['output_dir'] = SCRIPT_DIR

    exp_cfg = cfg.setdefault('experiment', {})
    lr = cfg['train']['optimizer']['lr']
    exp_name = exp_cfg.get('name') or f"dml_cremad_seed{cfg['seed']}_lr{lr}"
    exp_cfg['name'] = exp_name
    save_root = cfg.get('save_dir') or os.path.join(
        cfg.get('output_dir', SCRIPT_DIR),
        'savepath',
        cfg['dataset']['dataset_name'],
    )
    cfg['save_dir'] = save_root
    savedir = os.path.join(save_root, exp_name)
    cfg['run_dir'] = savedir
    cfg['log_name'] = "training.log"
    os.makedirs(savedir, exist_ok=True)

    # ----- SET SEED -----
    setup_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.get("gpu_id", 0))

    # ----- SET LOGGER -----
    local_rank = cfg['train'].get('local_rank', 0)
    logger, log_file, exp_id = create_logger(cfg, local_rank)

    # ----- SET DATALOADER -----
    train_dataset = CramedDataset(cfg, mode='train')
    test_dataset = CramedDataset(cfg, mode='test')

    num_samples = len(train_dataset)

    split_indices = torch.randperm(num_samples).tolist()
    val_size = min(cfg['train']['val_size'], max(1, num_samples - 1))
    calib_size = resolve_train_split_calib_size(num_samples - val_size, cfg.get('calib_size', 0))
    val_indices = split_indices[:val_size]
    val_dataset = Subset(train_dataset, val_indices)

    calib_indices = split_indices[val_size:val_size + calib_size]
    calib_dataset = Subset(train_dataset, calib_indices)

    train_indices = split_indices[val_size + calib_size:]
    train_dataset_sub = Subset(train_dataset, train_indices)

    logger.info(
        f'Train samples: {len(train_dataset_sub)}, '
        f'Val samples: {len(val_dataset)}, '
        f'Calibration samples: {len(calib_dataset)}, '
        f'Test samples: {len(test_dataset)}'
    )

    train_loader = DataLoader(
        dataset=train_dataset_sub,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg['train']['val_batch_size'],
        shuffle=False,
        num_workers=cfg['train']['val_num_workers'],
        pin_memory=True
    )

    calib_loader = DataLoader(
        dataset=calib_dataset,
        batch_size=cfg['train']['val_batch_size'],
        shuffle=False,
        num_workers=cfg['train']['val_num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg['test']['batch_size'],
        shuffle=False,
        num_workers=cfg['test']['num_workers'],
        pin_memory=True
    )

    # ----- MODEL -----
    model = DMLClassifier(config=cfg)
    model = model.cuda()
    model.apply(weight_init)

    # ----- OPTIMIZER & SCHEDULER -----
    lr = cfg['train']['optimizer']['lr']
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=cfg['train']['optimizer']['momentum'],
        weight_decay=cfg['train']['optimizer']['wc']
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        cfg['train']['lr_scheduler']['patience'],
        0.1
    )

    # ----- TRAINING LOOP -----
    n_no_improve = 0
    early_stopping_patience = cfg['train']['early_stopping_patience']
    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(f'Epoch {epoch} is pending...')
        calib_logits, calib_labels = collect_multimodal_logits(calib_loader, model)
        epoch_conformal_calibration = calibrate_conformal(
            calib_logits,
            calib_labels,
            cfg['conformal_alpha'],
        )
        model = train_audio_video(
            epoch,
            train_loader,
            model,
            optimizer,
            logger,
            cfg,
            conformal_thresholds=epoch_conformal_calibration["thresholds"],
        )
        previous_best_acc = best_acc
        acc, acc_a, acc_v = val(epoch, val_loader, model, logger)
        is_improvement = acc > previous_best_acc

        # Save best model
        if is_improvement and acc > 0:
            n_no_improve = 0
            save_path = os.path.join(savedir, 'model_best_clean.pt')
            torch.save(model.state_dict(), save_path)
            logger.info(f'Model saved to {save_path}')
        else:
            n_no_improve += 1

        scheduler.step()
        if n_no_improve >= early_stopping_patience:
            logger.info(
                f"No improvement for {early_stopping_patience} epochs. Stopping early."
            )
            break

    # ----- FINAL SUMMARY -----
    logger.info('=' * 60)
    logger.info(f'Training complete. Best epoch: {best_epoch}, Best acc: {best_acc:.4f}')
    logger.info(f'Best metrics: {best_log}')
    logger.info('=' * 60)

    # ----- ROBUSTNESS EVALUATION -----
    logger.info("Loading best model for robustness evaluation...")
    best_ckpt_path = os.path.join(savedir, 'model_best_clean.pt')

    eval_model = DMLClassifier(config=cfg).cuda()
    if os.path.isfile(best_ckpt_path):
        eval_model.load_state_dict(torch.load(best_ckpt_path, map_location='cuda'))
        logger.info(f"Loaded best model from {best_ckpt_path}")
    else:
        eval_model.load_state_dict(model.state_dict())
        logger.info("No best checkpoint found, using current model state.")
    eval_model.eval()

    calib_logits, calib_labels = collect_multimodal_logits(calib_loader, eval_model)
    conformal_calibration = calibrate_conformal(
        calib_logits,
        calib_labels,
        cfg['conformal_alpha'],
    )
    conformal_results = {
        "alpha": float(cfg['conformal_alpha']),
        "tau": float(cfg["uncertainty_tau"]),
        "calibration": {
            "source": "train_split",
            "size": int(conformal_calibration["n_calibration"]),
        },
        "thresholds": conformal_calibration["thresholds"],
        "robustness": {},
    }

    # Define 5 robustness scenarios
    test_scenarios = {
        "Clean Test": {'noise_type': None, 'noise_level': 0},
        "Salt & Pepper (Lvl 5.0)": {'noise_type': 'salt_pepper', 'noise_level': 5.0},
        "Salt & Pepper (Lvl 10.0)": {'noise_type': 'salt_pepper', 'noise_level': 10.0},
        "Gaussian (Lvl 5.0)": {'noise_type': 'gaussian', 'noise_level': 5.0},
        "Gaussian (Lvl 10.0)": {'noise_type': 'gaussian', 'noise_level': 10.0},
    }

    final_results = {}
    for scenario_name, noise_cfg in test_scenarios.items():
        logger.info(f"Evaluating scenario: {scenario_name} ...")

        noised_dataset = CramedDatasetNoised(
            cfg,
            mode='test',
            noise_type=noise_cfg['noise_type'],
            noise_level=noise_cfg['noise_level'],
        )
        noised_loader = DataLoader(
            dataset=noised_dataset,
            batch_size=cfg['test']['batch_size'],
            shuffle=False,
            num_workers=cfg['test']['num_workers'],
            pin_memory=True,
        )

        scenario_logits, scenario_labels = collect_multimodal_logits(noised_loader, eval_model)
        pred_list = scenario_logits["fused"].argmax(axis=1)
        acc = sum(1 for x, y in zip(scenario_labels, pred_list) if x == y) / len(scenario_labels)
        final_results[scenario_name] = float(acc)
        conformal_results["robustness"][scenario_name] = evaluate_conformal(
            scenario_logits,
            scenario_labels,
            conformal_calibration["thresholds"],
            tau=cfg['uncertainty_tau'],
        )
        logger.info(f"  {scenario_name}: Acc = {acc:.4f}")

    # Print formatted results table
    logger.info("\n" + "=" * 60)
    logger.info(f"{'FINAL ROBUSTNESS EVALUATION RESULTS':^60}")
    logger.info("=" * 60)
    logger.info(f"| {'Test Scenario':<35} | {'Accuracy':<18} |")
    logger.info("|" + "-" * 37 + "|" + "-" * 20 + "|")
    for name, score in final_results.items():
        logger.info(f"| {name:<35} | {score:>18.4f} |")
    logger.info("=" * 60 + "\n")

    # Save final_results.json
    result_file = os.path.join(savedir, "final_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_clean_model": {
                    "epoch": int(best_epoch),
                    "clean_acc": float(final_results.get("Clean Test", 0.0)),
                },
                "robustness": final_results,
                "conformal": conformal_results,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )
    logger.info(f"Saved final results to {result_file}")

    summary_path = os.path.join(os.path.dirname(savedir), "all_experiments.json")
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": cfg['dataset']['dataset_name'],
        "name": exp_cfg['name'],
        "note": exp_cfg.get('note', ''),
        "seed": cfg['seed'],
        "lr": cfg['train']['optimizer']['lr'],
        "batch_sz": cfg['train']['batch_size'],
        "ib_beta": cfg.get("ib_beta", 1e-3),
        "ib_eps_scale": cfg.get("ib_eps_scale", 1.0),
        "conformal_alpha": cfg.get("conformal_alpha", 0.1),
        "uncertainty_tau": cfg.get("uncertainty_tau", 1.0),
        "calib_size": cfg.get("calib_size", 0),
        "max_epochs": cfg['train']['epoch_dict'],
        "patience": early_stopping_patience,
        "best_clean_epoch": int(best_epoch),
        "best_clean_acc": float(final_results.get("Clean Test", 0.0)),
        "robustness": final_results,
        "conformal": conformal_results,
        "savedir": savedir,
    }
    append_experiment_record(summary_path, record)
    logger.info(f"Appended experiment record to {summary_path}")

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


def train_audio_video(epoch, train_loader, model, optimizer, logger, cfg):
    """单 epoch 训练循环（决策级融合 + 三路 CE 损失之和）。

    损失 = CE(fused, target) + CE(audio, target) + CE(video, target)
    """
    model.train()
    tl = Averager()
    tl_fused = Averager()
    tl_audio = Averager()
    tl_video = Averager()
    tl_ib = Averager()
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

        fused_logits, audio_logits, video_logits, _, _, ib_loss = model(spectrogram, image)
        loss, loss_parts = information_bottleneck_classification_loss(
            criterion,
            fused_logits,
            audio_logits,
            video_logits,
            y,
            ib_loss,
            ib_beta,
        )

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



if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join(SCRIPT_DIR, 'data', 'crema.json'),
    )
    parser.add_argument('--early_stop_patience', type=int, default=None)
    parser.add_argument('--early_stop_min_delta', type=float, default=None)
    args = parser.parse_args()

    cfg = config

    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)
    if args.early_stop_patience is not None:
        cfg['train']['early_stop_patience'] = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        cfg['train']['early_stop_min_delta'] = args.early_stop_min_delta

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

    val_indices = torch.randperm(num_samples)[:cfg['train']['val_size']]
    val_dataset = Subset(train_dataset, val_indices)

    train_indices = torch.tensor([i for i in range(num_samples) if i not in val_indices.tolist()])
    train_dataset_sub = Subset(train_dataset, train_indices)

    logger.info(f'Train samples: {len(train_dataset_sub)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}')

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
    early_stop_counter = 0
    early_stop_best_acc = -float("inf")
    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(f'Epoch {epoch} is pending...')
        model = train_audio_video(epoch, train_loader, model, optimizer, logger, cfg)
        acc, acc_a, acc_v = val(epoch, test_loader, model, logger)

        # Save best model
        if acc > early_stop_best_acc + cfg['train']['early_stop_min_delta']:
            early_stop_best_acc = acc
            early_stop_counter = 0
            save_path = os.path.join(savedir, 'model_best_clean.pt')
            if acc > 0:
                torch.save(model.state_dict(), save_path)
                logger.info(f'Model saved to {save_path}')
        else:
            early_stop_counter += 1
            logger.info(
                f"No clean accuracy improvement for {early_stop_counter}/"
                f"{cfg['train']['early_stop_patience']} epochs."
            )

        scheduler.step()
        if (
            cfg['train']['early_stop_patience'] > 0
            and early_stop_counter >= cfg['train']['early_stop_patience']
        ):
            logger.info(
                f"Stopping early at epoch {epoch}: clean accuracy did not improve "
                f"by at least {cfg['train']['early_stop_min_delta']} for "
                f"{cfg['train']['early_stop_patience']} epochs."
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

        # Evaluate
        eval_model.eval()
        pred_list = []
        label_list = []

        with torch.no_grad():
            for spectrogram, image, y in tqdm(noised_loader, desc=scenario_name):
                label_list.extend(torch.argmax(y, dim=1).tolist())
                image = image.float().cuda()
                spectrogram = spectrogram.unsqueeze(1).float().cuda()

                fused_logits, _, _, _, _, _ = eval_model(spectrogram, image)
                pred = F.softmax(fused_logits, dim=1).argmax(dim=1)
                pred_list.extend(pred.cpu().tolist())

        acc = sum(1 for x, y in zip(label_list, pred_list) if x == y) / len(label_list)
        final_results[scenario_name] = float(acc)
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
        "max_epochs": cfg['train']['epoch_dict'],
        "best_clean_epoch": int(best_epoch),
        "best_clean_acc": float(final_results.get("Clean Test", 0.0)),
        "robustness": final_results,
        "savedir": savedir,
    }
    append_experiment_record(summary_path, record)
    logger.info(f"Appended experiment record to {summary_path}")

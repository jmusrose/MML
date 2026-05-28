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
from sklearn.metrics import f1_score, average_precision_score

from data.template import config
from dataset.CREMA import CramedDataset
from dataset.CREMA_noised import CramedDatasetNoised
from model.DMLClassifier import DMLClassifier
from utils.utils import (
    create_logger,
    Averager,
    deep_update_dict,
)
from utils.tools import weight_init, compute_mAP, setup_seed


def train_audio_video(epoch, train_loader, model, optimizer, logger):
    """单 epoch 训练循环（决策级融合 + 三路 CE 损失之和）。

    损失 = CE(fused, target) + CE(audio, target) + CE(video, target)
    """
    model.train()
    tl = Averager()
    criterion = nn.CrossEntropyLoss().cuda()

    for step, (spectrogram, image, y) in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch}")):
        image = image.float().cuda()
        y = y.cuda()
        spectrogram = spectrogram.unsqueeze(1).float().cuda()

        optimizer.zero_grad()

        fused_logits, audio_logits, video_logits, _, _ = model(spectrogram, image)

        loss_fused = criterion(fused_logits, y)
        loss_audio = criterion(audio_logits, y)
        loss_video = criterion(video_logits, y)
        loss = loss_fused + loss_audio + loss_video

        # NaN detection
        if torch.isnan(loss).any():
            logger.info(
                f"NaN detected at step {step}: "
                f"loss_fused={loss_fused.item()}, "
                f"loss_audio={loss_audio.item()}, "
                f"loss_video={loss_video.item()}"
            )

        loss.backward()
        optimizer.step()

        tl.add(loss.item())

    loss_ave = tl.item()
    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(f'Epoch {epoch}: Average Training Loss: {loss_ave:.4f}')
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

            fused_logits, audio_logits, video_logits, _, _ = model(spectrogram, image)

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
    args = parser.parse_args()

    cfg = config

    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)

    # Anchor output_dir to the script directory if user left it as '.'
    # so logs/checkpoints land next to this script regardless of cwd.
    if cfg.get('output_dir', '.') in ('.', './', ''):
        cfg['output_dir'] = SCRIPT_DIR

    # ----- SET SEED -----
    setup_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # ----- SET LOGGER -----
    local_rank = cfg['train'].get('local_rank', 0)
    logger, log_file, exp_id = create_logger(cfg, local_rank)

    # ----- SET DATALOADER -----
    train_dataset = CramedDataset(cfg, mode='train')
    test_dataset = CramedDataset(cfg, mode='test')

    num_samples = len(train_dataset)

    val_indices = torch.randperm(num_samples)[:16]
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
        batch_size=16,
        shuffle=False,
        num_workers=8,
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
    savedir = os.path.join(cfg.get('output_dir', '.'), cfg['dataset']['dataset_name'], 'checkpoints')
    os.makedirs(savedir, exist_ok=True)

    for epoch in range(cfg['train']['epoch_dict']):
        logger.info(f'Epoch {epoch} is pending...')
        model = train_audio_video(epoch, train_loader, model, optimizer, logger)
        acc, acc_a, acc_v = val(epoch, test_loader, model, logger)

        # Save best model
        if acc >= best_acc and acc > 0:
            save_path = os.path.join(savedir, 'model_best.pt')
            torch.save(model.state_dict(), save_path)
            logger.info(f'Model saved to {save_path}')

        scheduler.step()

    # ----- FINAL SUMMARY -----
    logger.info('=' * 60)
    logger.info(f'Training complete. Best epoch: {best_epoch}, Best acc: {best_acc:.4f}')
    logger.info(f'Best metrics: {best_log}')
    logger.info('=' * 60)

    # ----- ROBUSTNESS EVALUATION -----
    logger.info("Loading best model for robustness evaluation...")
    best_ckpt_path = os.path.join(savedir, 'model_best.pt')

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

                fused_logits, _, _, _, _ = eval_model(spectrogram, image)
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
        )
    logger.info(f"Saved final results to {result_file}")

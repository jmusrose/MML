#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DML_v1/RGB/DML_nyu.py

NYU Depth V2 主入口脚本。

本文件定义命令行配置、优化器/调度器构造工具、多标签 mAP 助手、单 epoch 训练
循环 ``train_rgbd`` / 验证循环 ``val_rgbd``，以及完整的训练-保存-鲁棒评估
流水线 ``main``。

风格参考：``CPSC_RGB/CPSC_nyu.py``。
"""

import argparse
import copy
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.additional_transform import AddGaussianNoise, AddSaltPepperNoise
from data.aligned_conc_dataset import AlignedConcDataset
from data.aligned_conc_dataset_noised import AlignedConcDataset as AlignedConcDatasetNoised
from models.dml_classifier_nyu import Classifier
from tool.loss import information_bottleneck_classification_loss
from utils.conformal import (
    calibrate_conformal,
    conformal_uncertainty_from_logits,
    evaluate_conformal,
)
from utils.logger import create_logger
from utils.utils import Averager, append_experiment_record, set_seed


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)


def get_args(parser: argparse.ArgumentParser) -> None:
    """注册 NYU Depth V2 主入口的全部命令行参数。

    所有参数及默认值与 design.md 的参数表保持一致；``--n_classes`` 默认为
    19，``--img_embed_pool_type`` 受限为 ``{"max", "avg"}``。
    """
    parser.add_argument("--batch_sz", type=int, default=64)
    parser.add_argument("--data_path", type=str, default=os.path.join(_REPO_ROOT, "datasets_shared", "nyud2_trainvaltest"))
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2.4e-4)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--savedir", type=str, default=os.path.join(_THIS_DIR, "savepath", "nyud"))
    parser.add_argument("--name", type=str, default="s")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_classes", type=int, default=19)
    parser.add_argument("--img_hidden_sz", type=int, default=512)
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--ib_beta", type=float, default=1e-3)
    parser.add_argument("--ib_eps_scale", type=float, default=1.0)
    parser.add_argument("--conformal_alpha", type=float, default=0.1)
    parser.add_argument("--uncertainty_tau", type=float, default=1.0)
    parser.add_argument(
        "--calib_size",
        type=int,
        default=0,
        help="共形校准样本数；NYU<=0时使用完整val，SUN<=0时默认从train划20%",
    )
    parser.add_argument(
        "--img_embed_pool_type",
        type=str,
        default="avg",
        choices=["max", "avg"],
    )
    parser.add_argument(
        "--CONTENT_MODEL_PATH",
        type=str,
        default=os.path.join(_THIS_DIR, "checkpoint", "resnet18_pretrained.pth"),
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="本次实验备注，会随同最终结果追加到 all_experiments.json",
    )


def get_optimizer(model, args):
    """Adam 优化器，``lr=args.lr``、``weight_decay=1e-5``（与 CPSC_RGB 一致）。"""
    return optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)


def get_scheduler(optimizer, args):
    """ReduceLROnPlateau 调度器：``mode="max"``，由 ``args.lr_patience`` /
    ``args.lr_factor`` 控制，``verbose=True``。"""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=args.lr_patience,
        factor=args.lr_factor,
        verbose=True,
    )


def compute_mAP(outputs, labels):
    """多标签平均 AP（mean Average Precision）助手。

    参数
    ----
    outputs : Tensor[B, C]
        模型预测分数 / logits（按列对应每一类）。
    labels : Tensor[B, C]
        多标签 one-hot ground truth（与 ``outputs`` 同形状）。

    返回
    ----
    float
        所有类别 AP 的算术平均值。

    备注
    ----
    本助手与 ``CPSC_RGB/CPSC_nyu.py`` 中的实现等价，主要用于将来支持多标签
    评估。NYU/SUN 当前 baseline 使用单标签准确率（accuracy_score），此函数
    仅作扩展接口保留。
    """
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    aps = []
    for i in range(y_true.shape[1]):
        aps.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return float(np.mean(aps))


def _batch_conformal_ib_betas(rgb_out, depth_out, conformal_thresholds, args):
    if conformal_thresholds is None:
        return None, {}

    uncertainties = conformal_uncertainty_from_logits(
        {
            "rgb": rgb_out.detach().cpu().numpy(),
            "depth": depth_out.detach().cpu().numpy(),
        },
        conformal_thresholds,
    )
    ib_betas = {
        "rgb": args.ib_beta * (1.0 + uncertainties["rgb"]),
        "depth": args.ib_beta * (1.0 + uncertainties["depth"]),
    }
    return ib_betas, uncertainties


def train_rgbd(epoch, train_loader, model, optimizer, logger, args, conformal_thresholds=None):
    """单 epoch 训练循环（决策级融合 + 三路 CE 损失之和）。

    参数
    ----
    epoch : int
        当前 epoch 编号（仅用于日志与 tqdm 描述）。
    train_loader : torch.utils.data.DataLoader
        训练集 DataLoader，每个 batch 为字典 ``{'A', 'B', 'label', ...}``。
    model : nn.Module
        ``models.dml_classifier_nyu.Classifier`` 实例（已 ``.cuda()``）。
    optimizer : torch.optim.Optimizer
        优化器（典型为 Adam）。
    logger : logging.Logger
        训练日志器，epoch 结束时写入 ``Total Loss``。
    args : argparse.Namespace
        命令行配置，本函数当前未直接使用，但保留接口以便后续扩展。

    返回
    ----
    nn.Module
        训练后的同一 ``model``（in-place 更新参数后返回，方便链式调用）。

    设计说明
    --------
    - 损失为三路 CrossEntropy 之和：``CE(both) + CE(rgb) + CE(depth)``，与
      design.md 的决策级融合训练目标一致（Requirement 6.2）。
    - NaN 检测：若 ``loss`` 为 NaN 或 ``loss <= 0``，打印 step 编号与各分项
      损失值，但不抛异常（Requirement 12.1）。
    - 损失累计使用 ``Averager``，epoch 末通过 ``logger.info`` 输出
      ``f'Epoch {epoch}: Total Loss: {loss:.4f}'``（Requirement 6.4）。
    """
    model.train()
    #sadasd
    tl = Averager()
    tl_both = Averager()
    tl_rgb = Averager()
    tl_depth = Averager()
    tl_ib = Averager()
    tl_ib_rgb = Averager()
    tl_ib_depth = Averager()
    tl_beta_rgb = Averager()
    tl_beta_depth = Averager()
    tl_uncertainty_rgb = Averager()
    tl_uncertainty_depth = Averager()
    correct_both = 0
    correct_rgb = 0
    correct_depth = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss().cuda()

    for step, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch}")):
        rgb = batch['A'].cuda()
        depth = batch['B'].cuda()
        tgt = batch['label'].cuda()

        optimizer.zero_grad()

        both_out, rgb_out, depth_out, _, _, ib_losses = model(rgb, depth)
        ib_betas, uncertainties = _batch_conformal_ib_betas(
            rgb_out,
            depth_out,
            conformal_thresholds,
            args,
        )
        loss, loss_parts = information_bottleneck_classification_loss(
            criterion,
            both_out,
            rgb_out,
            depth_out,
            tgt,
            ib_losses,
            args.ib_beta,
            ib_betas=ib_betas,
        )
        if uncertainties:
            loss_parts["uncertainty_rgb"] = uncertainties["rgb"]
            loss_parts["uncertainty_depth"] = uncertainties["depth"]

        if torch.isnan(loss).any() or loss <= 0:
            print(f"NaN detected at step {step}")
            print(
                f"loss_both: {loss_parts['both'].item()}, "
                f"loss_rgb: {loss_parts['rgb'].item()}, "
                f"loss_depth: {loss_parts['depth'].item()}, "
                f"loss_ib: {loss_parts['ib'].item()}"
            )

        loss.backward()
        optimizer.step()

        tl.add(loss.item())
        tl_both.add(loss_parts['both'].item())
        tl_rgb.add(loss_parts['rgb'].item())
        tl_depth.add(loss_parts['depth'].item())
        tl_ib.add(loss_parts['ib'].item())
        if "ib_rgb" in loss_parts:
            tl_ib_rgb.add(loss_parts["ib_rgb"].item())
            tl_ib_depth.add(loss_parts["ib_depth"].item())
            tl_beta_rgb.add(loss_parts["beta_rgb"])
            tl_beta_depth.add(loss_parts["beta_depth"])
            tl_uncertainty_rgb.add(loss_parts["uncertainty_rgb"])
            tl_uncertainty_depth.add(loss_parts["uncertainty_depth"])
        with torch.no_grad():
            correct_both += (torch.argmax(both_out, dim=1) == tgt).sum().item()
            correct_rgb += (torch.argmax(rgb_out, dim=1) == tgt).sum().item()
            correct_depth += (torch.argmax(depth_out, dim=1) == tgt).sum().item()
            total_samples += tgt.size(0)

    loss = tl.item()
    loss_both_ave = tl_both.item()
    loss_rgb_ave = tl_rgb.item()
    loss_depth_ave = tl_depth.item()
    loss_ib_ave = tl_ib.item()
    loss_ib_rgb_ave = tl_ib_rgb.item()
    loss_ib_depth_ave = tl_ib_depth.item()
    beta_rgb_ave = tl_beta_rgb.item()
    beta_depth_ave = tl_beta_depth.item()
    uncertainty_rgb_ave = tl_uncertainty_rgb.item()
    uncertainty_depth_ave = tl_uncertainty_depth.item()
    acc_both = correct_both / total_samples if total_samples else 0.0
    acc_rgb = correct_rgb / total_samples if total_samples else 0.0
    acc_depth = correct_depth / total_samples if total_samples else 0.0
    logger.info(
        f'Epoch {epoch}: Total Loss: {loss:.4f}, '
        f'loss_both:{loss_both_ave:.4f}, '
        f'loss_rgb:{loss_rgb_ave:.4f}, '
        f'loss_depth:{loss_depth_ave:.4f}, '
        f'loss_ib:{loss_ib_ave:.4f}, '
        f'loss_ib_rgb:{loss_ib_rgb_ave:.4f}, '
        f'loss_ib_depth:{loss_ib_depth_ave:.4f}, '
        f'beta_rgb:{beta_rgb_ave:.6f}, '
        f'beta_depth:{beta_depth_ave:.6f}, '
        f'uncertainty_rgb:{uncertainty_rgb_ave:.4f}, '
        f'uncertainty_depth:{uncertainty_depth_ave:.4f}, '
        f'acc_both:{acc_both:.4f}, '
        f'acc_rgb:{acc_rgb:.4f}, '
        f'acc_depth:{acc_depth:.4f}'
    )
    return model


def val_rgbd(epoch, val_loader, model, logger, args):
    """单 epoch 验证循环（决策级融合融合分类准确率）。

    参数
    ----
    epoch : int
        当前 epoch 编号；约定 ``epoch == -1`` 表示最终鲁棒性评估调用，此时
        不打印 epoch 级日志（Requirement 7.3）。
    val_loader : torch.utils.data.DataLoader
        验证 / 测试 DataLoader,每个 batch 为字典 ``{'A', 'B', 'label', ...}``。
    model : nn.Module
        ``models.dml_classifier_nyu.Classifier`` 实例(已 ``.cuda()``)。
    logger : logging.Logger
        日志器,仅在 ``epoch != -1`` 时写入 ``Clean - Acc``。
    args : argparse.Namespace
        命令行配置,本函数当前未直接使用,但保留接口以便后续扩展。

    返回
    ----
    float
        ``sklearn.metrics.accuracy_score`` 计算得到的融合分支准确率。

    设计说明
    --------
    - 在 ``model.eval()`` 与 ``torch.no_grad()`` 上下文中遍历 loader,只取
      ``Classifier`` 5 元组的第一个元素 ``both_out`` 作为融合分支预测
      (Requirement 7.1)。
    - 当 ``epoch != -1`` 时,通过 ``logger.info`` 输出
      ``f'Epoch {epoch}: Clean - Acc: {acc:.4f}'``;``epoch == -1`` 时跳过该
      日志,以避免最终鲁棒性表格被多余行干扰(Requirement 7.3)。
    """
    model.eval()
    pred_list_fusion = []
    label_list = []

    with torch.no_grad():
        for batch in val_loader:
            rgb = batch['A'].cuda()
            depth = batch['B'].cuda()
            tgt = batch['label'].cuda()

            label_list.extend(tgt.cpu().tolist())

            both_out, _, _, _, _, _ = model(rgb, depth)
            pred_fusion = both_out.argmax(dim=1)
            pred_list_fusion.extend(pred_fusion.cpu().tolist())

    acc = accuracy_score(label_list, pred_list_fusion)
    if epoch != -1:
        logger.info(f'Epoch {epoch}: Val - Acc: {acc:.4f}')
    return acc


def collect_multimodal_logits(loader, model):
    model.eval()
    logits = {"fusion": [], "rgb": [], "depth": []}
    labels = []

    with torch.no_grad():
        for batch in loader:
            rgb = batch['A'].cuda()
            depth = batch['B'].cuda()
            tgt = batch['label'].cuda()

            both_out, rgb_out, depth_out, _, _, _ = model(rgb, depth)
            logits["fusion"].append(both_out.detach().cpu().numpy())
            logits["rgb"].append(rgb_out.detach().cpu().numpy())
            logits["depth"].append(depth_out.detach().cpu().numpy())
            labels.extend(tgt.cpu().tolist())

    return {
        modality: np.concatenate(values, axis=0)
        for modality, values in logits.items()
    }, np.asarray(labels, dtype=np.int64)


def build_nyu_dataloaders(args, train_transform, val_transform):
    """Build NYU train/val/test loaders from the dataset's explicit splits."""
    val_dataset = AlignedConcDataset(
        args,
        data_dir=os.path.join(args.data_path, 'val'),
        transform=val_transform,
    )
    calib_size = int(getattr(args, "calib_size", 0))
    if calib_size > 0:
        calib_dataset = Subset(val_dataset, list(range(min(calib_size, len(val_dataset)))))
    else:
        calib_dataset = val_dataset

    train_loader = DataLoader(
        AlignedConcDataset(
            args,
            data_dir=os.path.join(args.data_path, 'train'),
            transform=train_transform,
        ),
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
    )
    test_loader = DataLoader(
        AlignedConcDataset(
            args,
            data_dir=os.path.join(args.data_path, 'test'),
            transform=val_transform,
        ),
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
    )
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
    )
    return train_loader, val_loader, test_loader, calib_loader


def main():
    """NYU Depth V2 完整训练-保存-鲁棒评估流水线。

    流水线结构（与 design.md / requirements 2.5、7.2/7.4、8.x、12.1/12.2 一致）：

    1. 设置 ``CUDA_VISIBLE_DEVICES``、解析命令行参数、用 seed/lr 拼接
       ``args.name``，并 ``os.makedirs(args.savedir, exist_ok=True)``。
    2. 通过 ``set_seed`` 与 ``create_logger`` 完成可复现性 / 双输出日志的初始化。
    3. 构造 train/val/test transforms（mean/std 与 CPSC_RGB 完全一致）。
    4. 加载 train_dataset，用 ``torch.randperm`` 划出 4 个样本作 ``val_loader``，
       其余样本组成 ``train_loader``；test_loader 加载完整测试集。
    5. 实例化 ``Classifier(args).cuda()``、``get_optimizer``、``get_scheduler``。
    6. 训练循环：每 epoch ``train_rgbd → val_rgbd(test_loader) → 若 acc>best 保存
       ``model_best_clean.pt`` → ``scheduler.step(acc)``。
    7. 训练完成后加载 best 检查点；构造 5 个鲁棒性 loaders（Clean / SP-5 / SP-10 /
       G-5 / G-10）并依次调用 ``val_rgbd(-1, ...)``。
    8. 写日志表格 + ``final_results.json``（含 ``best_clean_model`` 与
       ``robustness`` 两个块，``indent=4``）。

    备注
    ----
    - 鲁棒性场景按 Requirement 8.5 / 8.6 配置：椒盐噪声等级 5.0 / 10.0 仅通过
      ``RandomApply`` 的 ``p`` 不同来区分（density 固定 0.10）；高斯噪声等级
      5.0 / 10.0 通过 ``variance`` 不同来区分（``p`` 固定 0.5）。
    - 同一 transform 同时作为 ``rgb_transform`` 与 ``depth_transform`` 传给
      ``AlignedConcDatasetNoised``，对应 Requirement 8.2。
    """
    # 1) 环境与命令行
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    parser = argparse.ArgumentParser(description="Train DML RGB-D Scene Recognition Model on NYU Depth V2")
    get_args(parser)
    args = parser.parse_args()

    args.name = f"dml_nyu_seed{args.seed}_lr{args.lr}"
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    # 2) 可复现性 + 日志
    set_seed(args.seed)
    logger = create_logger(os.path.join(args.savedir, "training.log"), args)

    # 3) Transforms（mean/std 与 CPSC_RGB 完全一致）
    mean = [0.4951, 0.3601, 0.4587]
    std = [0.1474, 0.1950, 0.1646]

    train_transforms = [
        transforms.Resize((args.LOAD_SIZE, args.LOAD_SIZE)),
        transforms.RandomCrop((args.FINE_SIZE, args.FINE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    val_transforms = [
        transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    # 4) 数据加载
    train_loader, val_loader, test_loader, calib_loader = build_nyu_dataloaders(
        args,
        transforms.Compose(train_transforms),
        transforms.Compose(val_transforms),
    )

    # 5) 模型 / 优化器 / 调度器
    model = Classifier(args).cuda()
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    # 6) 训练循环
    best_val_acc = 0.0
    best_val_epoch = 0
    best_val_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(args.max_epochs):
        logger.info(f'Epoch {epoch} training started...')
        calib_logits, calib_labels = collect_multimodal_logits(calib_loader, model)
        epoch_conformal_calibration = calibrate_conformal(
            calib_logits,
            calib_labels,
            args.conformal_alpha,
        )
        model = train_rgbd(
            epoch,
            train_loader,
            model,
            optimizer,
            logger,
            args,
            conformal_thresholds=epoch_conformal_calibration["thresholds"],
        )

        val_acc = val_rgbd(epoch, val_loader, model, logger, args)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_val_model_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': best_val_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                },
                os.path.join(args.savedir, 'model_best_val.pt'),
            )
            logger.info(
                f' *** NEW BEST VAL ACC *** Epoch {epoch} '
                f'| Val Acc: {best_val_acc:.4f}'
            )

        scheduler.step(val_acc)

    # 7) 鲁棒性评估：加载 best 检查点
    logger.info("Loading best model for final robust evaluation on Test sets...")
    best_ckpt_path = os.path.join(args.savedir, 'model_best_val.pt')
    clean_model = Classifier(args).cuda()
    if os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location='cuda')
        clean_model.load_state_dict(ckpt['model_state_dict'])
    else:
        # 训练过程中始终未刷新 best（例如 max_epochs=0），回退到内存中的初始/末次状态
        clean_model.load_state_dict(best_val_model_state)
    clean_model.eval()

    calib_logits, calib_labels = collect_multimodal_logits(calib_loader, clean_model)
    conformal_calibration = calibrate_conformal(
        calib_logits,
        calib_labels,
        args.conformal_alpha,
    )
    conformal_results = {
        "alpha": float(args.conformal_alpha),
        "tau": float(args.uncertainty_tau),
        "calibration": {
            "source": "val",
            "size": int(conformal_calibration["n_calibration"]),
        },
        "thresholds": conformal_calibration["thresholds"],
        "robustness": {},
    }

    # 鲁棒性场景的基础 transform 列表（Clean 不加噪声）
    clean_eval_transforms = [
        transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    # 椒盐噪声（Lvl 5.0：p=0.5；Lvl 10.0：p=1.0；density 固定 0.10）—— Req 8.5
    sp5_transforms = [
        transforms.Resize((256, 256)),
        transforms.RandomApply([AddSaltPepperNoise(density=0.10)], p=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    sp10_transforms = [
        transforms.Resize((256, 256)),
        transforms.RandomApply([AddSaltPepperNoise(density=0.10)], p=1.0),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    # 高斯噪声（Lvl 5.0：variance=5；Lvl 10.0：variance=10；p 固定 0.5）—— Req 8.6
    gs5_transforms = [
        transforms.Resize((256, 256)),
        transforms.RandomApply([AddGaussianNoise(mean=0.0, variance=5)], p=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    gs10_transforms = [
        transforms.Resize((256, 256)),
        transforms.RandomApply([AddGaussianNoise(mean=0.0, variance=10)], p=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    # 5 个 loader：同一 transform 同时作 rgb_transform / depth_transform（Req 8.2）
    test_data_dir = os.path.join(args.data_path, 'test')
    test_scenarios = {
        "Clean Test": DataLoader(
            AlignedConcDatasetNoised(
                args,
                data_dir=test_data_dir,
                rgb_transform=transforms.Compose(clean_eval_transforms),
                depth_transform=transforms.Compose(clean_eval_transforms),
            ),
            batch_size=args.batch_sz, shuffle=False, num_workers=args.n_workers,
        ),
        "Salt & Pepper (Lvl 5.0)": DataLoader(
            AlignedConcDatasetNoised(
                args,
                data_dir=test_data_dir,
                rgb_transform=transforms.Compose(sp5_transforms),
                depth_transform=transforms.Compose(sp5_transforms),
            ),
            batch_size=args.batch_sz, shuffle=False, num_workers=args.n_workers,
        ),
        "Salt & Pepper (Lvl 10.0)": DataLoader(
            AlignedConcDatasetNoised(
                args,
                data_dir=test_data_dir,
                rgb_transform=transforms.Compose(sp10_transforms),
                depth_transform=transforms.Compose(sp10_transforms),
            ),
            batch_size=args.batch_sz, shuffle=False, num_workers=args.n_workers,
        ),
        "Gaussian (Lvl 5.0)": DataLoader(
            AlignedConcDatasetNoised(
                args,
                data_dir=test_data_dir,
                rgb_transform=transforms.Compose(gs5_transforms),
                depth_transform=transforms.Compose(gs5_transforms),
            ),
            batch_size=args.batch_sz, shuffle=False, num_workers=args.n_workers,
        ),
        "Gaussian (Lvl 10.0)": DataLoader(
            AlignedConcDatasetNoised(
                args,
                data_dir=test_data_dir,
                rgb_transform=transforms.Compose(gs10_transforms),
                depth_transform=transforms.Compose(gs10_transforms),
            ),
            batch_size=args.batch_sz, shuffle=False, num_workers=args.n_workers,
        ),
    }

    final_results = {}
    for name, loader in test_scenarios.items():
        logger.info(f"Generating and Evaluating on dataset for: {name} ...")
        scenario_logits, scenario_labels = collect_multimodal_logits(loader, clean_model)
        pred_fusion = scenario_logits["fusion"].argmax(axis=1)
        acc = accuracy_score(scenario_labels, pred_fusion)
        final_results[name] = float(acc)
        conformal_results["robustness"][name] = evaluate_conformal(
            scenario_logits,
            scenario_labels,
            conformal_calibration["thresholds"],
            tau=args.uncertainty_tau,
        )

    # 8) 表格日志 + final_results.json
    logger.info("\n" + "=" * 60)
    logger.info(f"{'FINAL ROBUSTNESS EVALUATION RESULTS ON TEST SET':^60}")
    logger.info("=" * 60)
    logger.info(f"| {'Test Scenario':<35} | {'Score (Acc)':<18} |")
    logger.info("|" + "-" * 37 + "|" + "-" * 20 + "|")
    for name, score in final_results.items():
        logger.info(f"| {name:<35} | {score:>18.4f} |")
    logger.info("=" * 60 + "\n")

    result_file = os.path.join(args.savedir, "final_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_model": {
                    "epoch": int(best_val_epoch),
                    "val_acc": float(best_val_acc),
                    "clean_test_acc": float(final_results["Clean Test"]),
                },
                "robustness": final_results,
                "conformal": conformal_results,
            },
            f,
            indent=4,
        )

    logger.info(f"Saved final results to {result_file}")
    logger.info(
        f"Best Val Model: epoch={best_val_epoch}, "
        f"val_acc={best_val_acc:.4f}, "
        f"clean_test_acc={final_results['Clean Test']:.4f}"
    )

    # 9) 追加到该数据集的总实验记录（dirname(savedir) 即 ./savepath/nyud/）
    summary_path = os.path.join(os.path.dirname(args.savedir), "all_experiments.json")
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": "nyu",
        "name": args.name,
        "note": args.note,
        "seed": args.seed,
        "lr": args.lr,
        "ib_beta": args.ib_beta,
        "ib_eps_scale": args.ib_eps_scale,
        "uncertainty_tau": args.uncertainty_tau,
        "batch_sz": args.batch_sz,
        "max_epochs": args.max_epochs,
        "best_val_epoch": int(best_val_epoch),
        "best_val_acc": float(best_val_acc),
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

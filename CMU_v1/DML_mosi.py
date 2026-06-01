#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DML_v1/CMU_v1/DML_mosi.py

CMU-MOSI 主入口脚本。

本文件定义命令行配置、优化器/调度器构造工具、单 epoch 训练循环
``train_cmu`` / 验证循环 ``val_cmu``，以及完整的训练-保存-鲁棒评估
流水线 ``main``。

风格参考：``DML_v1/RGB_v1/DML_nyu.py``。MOSI 与 MOSEI 在训练循环层面
完全镜像，差异主要在主入口的默认 CLI 参数与 ``Classifier`` 来源。
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.additional_transform import AddFeatureGaussianNoise
from data.cmu_aligned_dataset import AlignedConcDataset
from data.cmu_aligned_dataset_noised import AlignedConcDataset as AlignedConcDatasetNoised
from models.dml_classifier_mosi import Classifier
from utils.logger import create_logger
from utils.metrics import compute_cmu_metrics
from utils.utils import Averager, append_experiment_record, set_seed


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)


def get_args(parser: argparse.ArgumentParser) -> None:
    """注册 CMU-MOSI 主入口的全部命令行参数。

    所有参数及默认值与 design.md 的参数表保持一致；``--pool_strategy``
    通过 ``choices=["last", "default"]`` 限定取值（Requirement 2.5）。

    共享参数（与 RGB_v1 同名）::

        --batch_sz, --data_path, --dropout, --lr, --lr_factor, --lr_patience,
        --max_epochs, --n_workers, --savedir, --name, --seed, --n_classes,
        --note

    CMU 专属参数::

        --dataset, --vision_dim, --audio_dim, --hidden_sz, --num_heads,
        --num_layers, --conv_kernel_size, --max_seq_len, --bert_model_name,
        --bert_max_len, --text_hidden_sz, --pool_strategy, --freeze_bert
    """
    # --- 共享参数（命名与 RGB_v1 完全一致） ---
    parser.add_argument("--batch_sz", type=int, default=32)
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(_REPO_ROOT, "datasets_shared", "mosi.pkl"),
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="非 BERT 部分（vision/audio encoder + 三路线性头）的学习率",
    )
    parser.add_argument(
        "--bert_lr",
        type=float,
        default=2e-5,
        help="BERT 微调专用学习率（HF 推荐值 2e-5 ~ 5e-5）",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="梯度裁剪阈值（按 L2 范数）；<= 0 时禁用",
    )
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument(
        "--savedir",
        type=str,
        default=os.path.join(_THIS_DIR, "savepath", "mosi"),
    )
    parser.add_argument("--name", type=str, default="s")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_classes",
        type=int,
        default=1,
        help="回归输出维度，baseline 固定为 1",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="本次实验备注，会随同最终结果追加到 all_experiments.json",
    )

    # --- CMU 专属参数 ---
    parser.add_argument("--dataset", type=str, default="mosi")
    parser.add_argument("--vision_dim", type=int, default=47)
    parser.add_argument("--audio_dim", type=int, default=74)
    parser.add_argument("--hidden_sz", type=int, default=50)
    parser.add_argument("--num_heads", type=int, default=5)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--conv_kernel_size", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--bert_max_len", type=int, default=50)
    parser.add_argument("--text_hidden_sz", type=int, default=768)
    parser.add_argument(
        "--pool_strategy",
        type=str,
        default="last",
        choices=["last", "default"],
    )
    parser.add_argument(
        "--freeze_bert",
        action="store_true",
        default=False,
        help="冻结 BERT 全部参数",
    )


def get_optimizer(model, args):
    """Adam 优化器，BERT 与非 BERT 部分使用分组学习率。

    BERT (`text_enc.bert`) 的 110M 参数对学习率非常敏感，HuggingFace 官方
    微调推荐 ``2e-5 ~ 5e-5``；vision/audio encoder 与三路线性头是从头训
    的小模型，需要更大学习率（``1e-3`` 量级）才能学得动。共享同一个
    ``lr`` 会让 BERT 过拟合且小网络欠拟合，因此按模块拆成两个 param
    group：

    - group 0 (BERT) → ``lr=args.bert_lr``
    - group 1 (其余) → ``lr=args.lr``

    ``ReduceLROnPlateau`` 会按 ``factor`` 同步缩放两组 lr，组间比例保持
    不变（HF 微调实践与 PyTorch 文档均允许此用法）。
    """
    bert_params = list(model.text_enc.bert.parameters())
    bert_param_ids = {id(p) for p in bert_params}
    other_params = [p for p in model.parameters() if id(p) not in bert_param_ids]
    return optim.Adam(
        [
            {"params": bert_params, "lr": args.bert_lr},
            {"params": other_params, "lr": args.lr},
        ],
        weight_decay=1e-5,
    )


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


def train_cmu(epoch, train_loader, model, optimizer, logger, args):
    """单 epoch 训练循环（决策级融合 + 四路 MSE 损失之和）。

    设计说明
    --------
    - 损失为四路 MSELoss（L2）之和:
      ``MSE(both) + MSE(vision) + MSE(audio) + MSE(text)``，每路 logit 形状
      为 ``(B, 1)``，与 batch label ``(B,)`` 计算 MSELoss 前先 ``squeeze(-1)``。
    - NaN 检测：若 ``loss`` 为 NaN 或 ``loss <= 0``，打印 step 编号与各分项
      损失值，但不抛异常（Requirement 7.6 / 13.1）。
    - 损失累计使用 ``Averager``，epoch 末通过 ``logger.info`` 输出
      ``f'Epoch {epoch}: Total Loss: {loss:.4f}'``（Requirement 7.4 / 7.5）。
    """
    model.train()

    tl = Averager()
    # criterion = nn.MSELoss().cuda()
    criterion = torch.nn.L1Loss().cuda()
    for step, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch}")):
        vision = batch['vision'].cuda()
        vision_mask = batch['vision_mask'].cuda()
        audio = batch['audio'].cuda()
        audio_mask = batch['audio_mask'].cuda()
        text_input_ids = batch['text_input_ids'].cuda()
        text_attention_mask = batch['text_attention_mask'].cuda()
        tgt = batch['label'].cuda().float()

        optimizer.zero_grad()

        both_output, vision_out, audio_out, text_out, _, _, _ = model(
            vision, vision_mask, audio, audio_mask,
            text_input_ids, text_attention_mask,
        )

        loss_both = criterion(both_output.squeeze(-1), tgt)
        loss_vision = criterion(vision_out.squeeze(-1), tgt)
        loss_audio = criterion(audio_out.squeeze(-1), tgt)
        loss_text = criterion(text_out.squeeze(-1), tgt)
        loss = loss_both + loss_vision + loss_audio + loss_text

        if torch.isnan(loss).any() or loss <= 0:
            print(f"NaN detected at step {step}")
            print(
                f"loss_both: {loss_both.item()}, "
                f"loss_vision: {loss_vision.item()}, "
                f"loss_audio: {loss_audio.item()}, "
                f"loss_text: {loss_text.item()}"
            )

        loss.backward()
        if getattr(args, "grad_clip", 0.0) and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        tl.add(loss.item())

    loss = tl.item()
    logger.info(f'Epoch {epoch}: Total Loss: {loss:.4f}')
    return model


def val_cmu(epoch, val_loader, model, logger, args):
    """单 epoch 验证循环（融合分支 5 指标评估）。

    设计说明
    --------
    - 在 ``model.eval()`` 与 ``torch.no_grad()`` 上下文中遍历 loader，
      只取 ``Classifier`` 7 元组的第一个元素 ``both_output`` 的
      ``squeeze(-1)`` 作为融合分支 scalar 预测（Requirement 8.1）。
    - 把全部 batch 的 ``(pred, label)`` 拼接为两个 1-D ``numpy.ndarray``，
      调用 ``compute_cmu_metrics`` 一次性得到 5 指标字典（Requirement 8.2）。
    - 当 ``epoch != -1`` 时通过 ``logger.info`` 输出 5 指标行；
      ``epoch == -1`` 时跳过该日志，避免最终鲁棒性表格被多余行干扰
      （Requirement 8.3 / 8.4）。
    """
    model.eval()
    pred_list = []
    label_list = []

    with torch.no_grad():
        for batch in val_loader:
            vision = batch['vision'].cuda()
            vision_mask = batch['vision_mask'].cuda()
            audio = batch['audio'].cuda()
            audio_mask = batch['audio_mask'].cuda()
            text_input_ids = batch['text_input_ids'].cuda()
            text_attention_mask = batch['text_attention_mask'].cuda()
            tgt = batch['label']

            label_list.extend(tgt.cpu().tolist())

            both_output, _, _, _, _, _, _ = model(
                vision, vision_mask, audio, audio_mask,
                text_input_ids, text_attention_mask,
            )
            pred = both_output.squeeze(-1)
            pred_list.extend(pred.cpu().tolist())

    pred_arr = np.asarray(pred_list, dtype=np.float64)
    label_arr = np.asarray(label_list, dtype=np.float64)
    metrics = compute_cmu_metrics(pred_arr, label_arr)

    if epoch != -1:
        logger.info(
            f"Epoch {epoch}: "
            f"MAE: {metrics['mae']:.4f} | "
            f"Corr: {metrics['corr']:.4f} | "
            f"Acc7: {metrics['acc7']:.4f} | "
            f"Acc2: {metrics['acc2']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )
    return metrics


def main():
    """CMU-MOSI 完整训练-保存-鲁棒评估流水线。

    流水线结构（与 design.md / requirements 1.2、2.x、6.x、7.x、8.x、9.x、
    10.4、11.1、11.3、12.1 一致）：

    1. 设置 ``CUDA_VISIBLE_DEVICES``、解析 CLI、用 seed/lr 拼接
       ``args.name = f"dml_mosi_seed{seed}_lr{lr}"``，并把 ``args.savedir``
       拼到子目录后 ``os.makedirs(..., exist_ok=True)``。
    2. ``set_seed(args.seed)`` 控制全部 PRNG，``create_logger`` 初始化双
       handler 日志器（控制台 + ``training.log``，UTF-8）。
    3. 通过 ``BertTokenizer.from_pretrained(args.bert_model_name)`` 构造
       tokenizer，加载三 split datasets（train/dev/test），按 RGB_v1 行为
       使用 test split 作训练期评估。
    4. 实例化 ``Classifier(args).cuda()``、``get_optimizer``、``get_scheduler``。
    5. 训练循环：每 epoch ``train_cmu → val_cmu(test_loader)``；以
       ``metrics['acc2']`` 为主指标维护 best 模型并保存
       ``model_best_clean.pt``；``scheduler.step(metrics['acc2'])``。
    6. 训练完成后加载 best 检查点（若不存在则回退到内存中的 best state），
       构造 5 个鲁棒场景 ``DML_NoisedDataset`` 与 DataLoader（Clean / Vision-
       Gauss-1 / Vision-Gauss-5 / Audio-Gauss-1 / Audio-Gauss-5），逐个调
       ``val_cmu(-1, ...)`` 收集 5 指标 dict。
    7. 通过 ``logger.info`` 写多列表格（``MAE | Corr | Acc7 | Acc2 | F1``）；
       把 ``best_clean_model`` + ``robustness`` 写入 ``final_results.json``
       （``indent=4, ensure_ascii=False``）。
    8. ``append_experiment_record`` 追加到该数据集的总结 JSON。
    """
    # 1) 环境与命令行
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    parser = argparse.ArgumentParser(
        description="Train DML CMU-MOSI Multimodal Sentiment Regression Model"
    )
    get_args(parser)
    args = parser.parse_args()
    # 主入口固定 dataset 标识
    args.dataset = "mosi"

    args.name = f"dml_mosi_seed{args.seed}_lr{args.lr}_blr{args.bert_lr}"
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    # 2) 可复现性 + 日志
    set_seed(args.seed)
    logger = create_logger(os.path.join(args.savedir, "training.log"), args)

    # 3) Tokenizer + 三 split datasets / DataLoaders
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)

    train_dataset = AlignedConcDataset(
        cfg=args, pkl_path=args.data_path, split="train", tokenizer=tokenizer,
    )
    dev_dataset = AlignedConcDataset(
        cfg=args, pkl_path=args.data_path, split="dev", tokenizer=tokenizer,
    )
    test_dataset = AlignedConcDataset(
        cfg=args, pkl_path=args.data_path, split="test", tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
    )
    # dev_loader 保留以便后续扩展（如 early stopping），不在 baseline 主路径上参与
    dev_loader = DataLoader(  # noqa: F841
        dev_dataset,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
    )

    # 4) 模型 / 优化器 / 调度器
    model = Classifier(args).cuda()
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    # 5) 训练循环（best 以 metrics['acc2'] 为主指标）
    best_clean_acc2 = -1.0
    best_clean_epoch = 0
    best_clean_metrics = {"mae": 0.0, "corr": 0.0, "acc7": 0.0, "acc2": 0.0, "f1": 0.0}
    best_clean_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(args.max_epochs):
        logger.info(f"Epoch {epoch} training started...")
        model = train_cmu(epoch, train_loader, model, optimizer, logger, args)

        metrics = val_cmu(epoch, test_loader, model, logger, args)

        if metrics["acc2"] > best_clean_acc2:
            best_clean_acc2 = float(metrics["acc2"])
            best_clean_epoch = epoch
            best_clean_metrics = {k: float(v) for k, v in metrics.items()}
            best_clean_model_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_clean_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": best_clean_metrics,
                    "clean_acc": best_clean_acc2,  # 兼容字段
                },
                os.path.join(args.savedir, "model_best_clean.pt"),
            )
            logger.info(
                f" *** NEW BEST CLEAN ACC2 *** Epoch {epoch} "
                f"| Acc2: {best_clean_acc2:.4f}"
            )

        scheduler.step(metrics["acc2"])

    # 6) 鲁棒性评估：加载 best 检查点
    logger.info("Loading best model for final robust evaluation on Test sets...")
    best_ckpt_path = os.path.join(args.savedir, "model_best_clean.pt")
    clean_model = Classifier(args).cuda()
    if os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cuda")
        clean_model.load_state_dict(ckpt["model_state_dict"])
    else:
        # 训练过程中始终未刷新 best（例如 max_epochs=0），回退到内存中的状态
        clean_model.load_state_dict(best_clean_model_state)
    clean_model.eval()

    # 5 个鲁棒场景：仅对一个目标模态加噪，另一模态保持 None
    test_scenarios_specs = [
        ("Clean Test", None, None),
        (
            "Vision Gaussian (Lvl 1.0)",
            AddFeatureGaussianNoise(mean=0.0, std=1.0, p=1.0),
            None,
        ),
        (
            "Vision Gaussian (Lvl 5.0)",
            AddFeatureGaussianNoise(mean=0.0, std=5.0, p=1.0),
            None,
        ),
        (
            "Audio Gaussian (Lvl 1.0)",
            None,
            AddFeatureGaussianNoise(mean=0.0, std=1.0, p=1.0),
        ),
        (
            "Audio Gaussian (Lvl 5.0)",
            None,
            AddFeatureGaussianNoise(mean=0.0, std=5.0, p=1.0),
        ),
    ]

    test_scenarios = {}
    for name, vt, at in test_scenarios_specs:
        ds = AlignedConcDatasetNoised(
            cfg=args,
            pkl_path=args.data_path,
            split="test",
            tokenizer=tokenizer,
            vision_transform=vt,
            audio_transform=at,
        )
        test_scenarios[name] = DataLoader(
            ds,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers,
        )

    final_results = {}
    for name, loader in test_scenarios.items():
        logger.info(f"Generating and Evaluating on dataset for: {name} ...")
        metrics = val_cmu(-1, loader, clean_model, logger, args)
        final_results[name] = {k: float(v) for k, v in metrics.items()}

    # 7) 表格日志 + final_results.json
    logger.info("\n" + "=" * 96)
    logger.info(f"{'FINAL ROBUSTNESS EVALUATION RESULTS ON TEST SET':^96}")
    logger.info("=" * 96)
    logger.info(
        f"| {'Test Scenario':<28} | {'MAE':>10} | {'Corr':>10} | "
        f"{'Acc7':>10} | {'Acc2':>10} | {'F1':>10} |"
    )
    logger.info(
        "|" + "-" * 30 + "|" + "-" * 12 + "|" + "-" * 12 + "|"
        + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|"
    )
    for name, m in final_results.items():
        logger.info(
            f"| {name:<28} | {m['mae']:>10.4f} | {m['corr']:>10.4f} | "
            f"{m['acc7']:>10.4f} | {m['acc2']:>10.4f} | {m['f1']:>10.4f} |"
        )
    logger.info("=" * 96 + "\n")

    result_file = os.path.join(args.savedir, "final_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_clean_model": {
                    "epoch": int(best_clean_epoch),
                    "metrics": best_clean_metrics,
                },
                "robustness": final_results,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(f"Saved final results to {result_file}")
    logger.info(
        f"Best Clean Model: epoch={best_clean_epoch}, "
        f"acc2={best_clean_metrics['acc2']:.4f}"
    )

    # 8) 追加到该数据集的总实验记录（dirname(savedir) 即 ./savepath/mosi/）
    summary_path = os.path.join(os.path.dirname(args.savedir), "all_experiments.json")
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": "mosi",
        "name": args.name,
        "note": args.note,
        "seed": args.seed,
        "lr": args.lr,
        "batch_sz": args.batch_sz,
        "max_epochs": args.max_epochs,
        "best_clean_epoch": int(best_clean_epoch),
        "best_clean_metrics": best_clean_metrics,
        "robustness": final_results,
        "savedir": args.savedir,
    }
    append_experiment_record(summary_path, record)
    logger.info(f"Appended experiment record to {summary_path}")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()

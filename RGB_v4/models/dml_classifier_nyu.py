"""
DML_v1/RGB/models/dml_classifier_nyu.py

NYU 数据集对应的 DML 多模态分类器（baseline，决策级融合）。

设计要点（与 design.md / requirements 4.x 严格一致）：

- 两个独立 `ImageEncoder` 实例：`self.rgbenc` / `self.depthenc`，参数完全不共享
- 不引入任何介于编码器与分类头之间的中间投影层
  （即没有 `unimodal_transform`、没有 MLP、没有非线性映射）
- 两个独立线性分类头：`self.rgb_clf` / `self.depth_clf`
  输入维度 = `img_hidden_sz × num_image_embeds`，输出维度 = `args.n_classes`
- forward 流程：encoder → `torch.flatten(start_dim=1)` → linear 分类头
- 决策级融合：`both_output = 0.5 * (rgb_out + depth_out)`
  除此之外不引入任何跨模态交互（无拼接、无注意力、无门控）
- 返回 5 元组：`(both_output, rgb_out, depth_out, rgb_uni, depth_uni)`
  其中 `rgb_uni` / `depth_uni` 即 `torch.flatten(encoder(x), start_dim=1)` 的结果
"""

import torch
import torch.nn as nn

from models.image import ImageEncoder


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args

        # 两路独立编码器（不共享权重）
        self.rgbenc = ImageEncoder(args)
        self.depthenc = ImageEncoder(args)

        # flatten 后特征维度 = num_image_embeds × img_hidden_sz
        d_in = args.img_hidden_sz * args.num_image_embeds

        # 两个独立线性分类头（无中间投影层）
        self.ib_eps_scale = getattr(args, "ib_eps_scale", 1.0)
        self.rgb_mu = nn.Linear(d_in, args.n_classes)
        self.rgb_logvar = nn.Linear(d_in, args.n_classes)
        self.depth_mu = nn.Linear(d_in, args.n_classes)
        self.depth_logvar = nn.Linear(d_in, args.n_classes)

    def _sample_logits(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + self.ib_eps_scale * eps * std

    @staticmethod
    def _kl_to_standard_normal(mu, logvar):
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=1).mean()

    def forward(self, rgb, depth):
        # encoder 输出形状: B × N × img_hidden_sz
        # flatten(start_dim=1) 后: B × (N × img_hidden_sz)
        rgb_uni = torch.flatten(self.rgbenc(rgb), start_dim=1)
        depth_uni = torch.flatten(self.depthenc(depth), start_dim=1)

        # 单模态 logit
        rgb_mu = self.rgb_mu(rgb_uni)
        rgb_logvar = self.rgb_logvar(rgb_uni)
        depth_mu = self.depth_mu(depth_uni)
        depth_logvar = self.depth_logvar(depth_uni)

        rgb_out = self._sample_logits(rgb_mu, rgb_logvar)
        depth_out = self._sample_logits(depth_mu, depth_logvar)
        ib_losses = {
            "rgb": self._kl_to_standard_normal(rgb_mu, rgb_logvar),
            "depth": self._kl_to_standard_normal(depth_mu, depth_logvar),
        }

        # 决策级融合（唯一融合方式）
        both_output = 0.5 * (rgb_out + depth_out)

        return both_output, rgb_out, depth_out, rgb_out, depth_out, ib_losses

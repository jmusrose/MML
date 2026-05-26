"""
DML_v1/RGB/models/dml_classifier_sun.py

SUN RGB-D 数据集对应的 DML 多模态分类器（baseline，决策级融合）。

baseline 阶段不在 NYU 与 SUN 之间做结构差异化，本文件与
`models/dml_classifier_nyu.py` 结构完全一致，仅文件归属语义不同。

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
        self.rgb_clf = nn.Linear(d_in, args.n_classes)
        self.depth_clf = nn.Linear(d_in, args.n_classes)

    def forward(self, rgb, depth):
        # encoder 输出形状: B × N × img_hidden_sz
        # flatten(start_dim=1) 后: B × (N × img_hidden_sz)
        rgb_uni = torch.flatten(self.rgbenc(rgb), start_dim=1)
        depth_uni = torch.flatten(self.depthenc(depth), start_dim=1)

        # 单模态 logit
        rgb_out = self.rgb_clf(rgb_uni)
        depth_out = self.depth_clf(depth_uni)

        # 决策级融合（唯一融合方式）
        both_output = 0.5 * (rgb_out + depth_out)

        return both_output, rgb_out, depth_out, rgb_uni, depth_uni

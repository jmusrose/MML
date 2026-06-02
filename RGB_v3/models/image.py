"""
DML_v1/RGB/models/image.py

ImageEncoder：基于 ResNet18 的图像编码器，用于 RGB 与 Depth 两路（参数不共享）。

- 优先加载 `args.CONTENT_MODEL_PATH` 指定的自定义预训练权重（若文件存在）
- 否则回退到 torchvision 提供的 ImageNet 预训练权重（兼容 PyTorch 1.x / 2.x）
- 去掉 ResNet18 最后的 fc 层，得到 (B, 512, 7, 7) 的特征图
- 接 `AdaptiveAvgPool2d` 或 `AdaptiveMaxPool2d`，pool 形状由 `args.num_image_embeds` 决定
- forward 末尾执行 `transpose(1, 2).contiguous()`，最终输出形状 `B × N × img_hidden_sz`
"""

import os

import torch
import torch.nn as nn
import torchvision
from torchvision import models  # noqa: F401  # 保留 import 以兼容 CPSC_RGB 风格


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args

        # 1) 优先加载用户指定的自定义预训练权重；2) 否则回退到 ImageNet 预训练
        content_path = getattr(args, "CONTENT_MODEL_PATH", None)
        if content_path and os.path.exists(content_path):
            model = torchvision.models.resnet18(pretrained=False)
            state_dict = torch.load(content_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"Loaded custom pretrained model from {content_path}")
        else:
            if content_path:
                print(
                    f"CONTENT_MODEL_PATH '{content_path}' not found, "
                    f"falling back to ImageNet pretrained weights."
                )
            try:
                # PyTorch 2.x 推荐写法
                from torchvision.models import ResNet18_Weights

                model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                print("Loaded ImageNet pretrained ResNet18 (PyTorch 2.x)")
            except (ImportError, AttributeError):
                # PyTorch 1.x 兼容写法
                model = torchvision.models.resnet18(pretrained=True)
                print("Loaded ImageNet pretrained ResNet18 (PyTorch 1.x)")

        # 去除最后的全连接分类头，仅保留特征提取主干
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)

        # 选择 pool 类型
        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        # 根据 num_image_embeds 选择 pool 形状
        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))
        else:
            raise ValueError(
                f"Unsupported num_image_embeds={args.num_image_embeds}; "
                f"expected one of {{1,2,3,4,5,6,7,8,9}}."
            )

    def forward(self, x):
        # B x 3 x H x W  -> B x 512 x 7 x 7  -> pool -> B x 512 x N
        # -> flatten(start_dim=2) -> B x 512 x N -> transpose(1,2) -> B x N x 512
        out = self.model(x)
        out = self.pool(out)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # B x N x img_hidden_sz

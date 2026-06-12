#!/usr/bin/env python3
"""Loss helpers for Food-101 information bottleneck training."""


def information_bottleneck_classification_loss(
    criterion,
    fused_logits,
    txt_logits,
    img_logits,
    target,
    ib_loss,
    ib_beta,
):
    """Three-branch CE plus beta-weighted information bottleneck KL."""
    loss_fused = criterion(fused_logits, target)
    loss_txt = criterion(txt_logits, target)
    loss_img = criterion(img_logits, target)
    weighted_ib = ib_beta * ib_loss
    loss = loss_fused + loss_txt + loss_img + weighted_ib
    return loss, {
        "fused": loss_fused,
        "txt": loss_txt,
        "img": loss_img,
        "ib": weighted_ib,
    }

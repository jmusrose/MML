#!/usr/bin/env python3
"""Loss helpers for Food-101 information bottleneck training."""


def information_bottleneck_classification_loss(
    criterion,
    fused_logits,
    txt_logits,
    img_logits,
    target,
    ib_losses,
    ib_beta,
    ib_betas=None,
):
    """Three-branch CE plus beta-weighted information bottleneck KL."""
    loss_fused = criterion(fused_logits, target)
    loss_txt = criterion(txt_logits, target)
    loss_img = criterion(img_logits, target)

    if isinstance(ib_losses, dict):
        if ib_betas is None:
            ib_betas = {
                "text": ib_beta,
                "image": ib_beta,
            }
        weighted_ib_txt = ib_betas["text"] * ib_losses["text"]
        weighted_ib_img = ib_betas["image"] * ib_losses["image"]
        weighted_ib = weighted_ib_txt + weighted_ib_img
    else:
        weighted_ib_txt = None
        weighted_ib_img = None
        weighted_ib = ib_beta * ib_losses

    loss = loss_fused + loss_txt + loss_img + weighted_ib
    parts = {
        "fused": loss_fused,
        "txt": loss_txt,
        "img": loss_img,
        "ib": weighted_ib,
    }
    if weighted_ib_txt is not None:
        parts["ib_text"] = weighted_ib_txt
        parts["ib_image"] = weighted_ib_img
        parts["beta_text"] = float(ib_betas["text"])
        parts["beta_image"] = float(ib_betas["image"])
    return loss, parts

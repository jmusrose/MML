#!/usr/bin/env python3
"""Loss helpers for CREMAD information bottleneck training."""


def information_bottleneck_classification_loss(
    criterion,
    fused_logits,
    audio_logits,
    video_logits,
    target,
    ib_losses,
    ib_beta,
    ib_betas=None,
):
    """Three-branch CE plus beta-weighted information bottleneck KL."""
    loss_fused = criterion(fused_logits, target)
    loss_audio = criterion(audio_logits, target)
    loss_video = criterion(video_logits, target)

    if isinstance(ib_losses, dict):
        if ib_betas is None:
            ib_betas = {
                "audio": ib_beta,
                "video": ib_beta,
            }
        weighted_ib_audio = ib_betas["audio"] * ib_losses["audio"]
        weighted_ib_video = ib_betas["video"] * ib_losses["video"]
        weighted_ib = weighted_ib_audio + weighted_ib_video
    else:
        weighted_ib_audio = None
        weighted_ib_video = None
        weighted_ib = ib_beta * ib_losses

    loss = loss_fused + loss_audio + loss_video + weighted_ib
    parts = {
        "fused": loss_fused,
        "audio": loss_audio,
        "video": loss_video,
        "ib": weighted_ib,
    }
    if weighted_ib_audio is not None:
        parts["ib_audio"] = weighted_ib_audio
        parts["ib_video"] = weighted_ib_video
        parts["beta_audio"] = float(ib_betas["audio"])
        parts["beta_video"] = float(ib_betas["video"])
    return loss, parts

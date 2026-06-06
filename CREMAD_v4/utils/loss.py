#!/usr/bin/env python3
"""Loss helpers for CREMAD information bottleneck training."""


def information_bottleneck_classification_loss(
    criterion,
    fused_logits,
    audio_logits,
    video_logits,
    target,
    ib_loss,
    ib_beta,
):
    """Three-branch CE plus beta-weighted information bottleneck KL."""
    loss_fused = criterion(fused_logits, target)
    loss_audio = criterion(audio_logits, target)
    loss_video = criterion(video_logits, target)
    weighted_ib = ib_beta * ib_loss
    loss = loss_fused + loss_audio + loss_video + weighted_ib
    return loss, {
        "fused": loss_fused,
        "audio": loss_audio,
        "video": loss_video,
        "ib": weighted_ib,
    }

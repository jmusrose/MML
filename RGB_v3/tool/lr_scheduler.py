"""tool/lr_scheduler.py

Placeholder module for custom learning-rate schedulers.

The current DML_v1/RGB baseline uses ``torch.optim.lr_scheduler.ReduceLROnPlateau``
configured directly inside the main entry scripts (``DML_nyu.py`` /
``DML_sun.py``), so nothing in this file is wired into the pipeline yet.

A ``WarmupMultiStepLR`` template is provided below for future experiments
(e.g. when switching to a step-based schedule with linear warmup).
"""

from bisect import bisect_right

import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Multi-step LR schedule with optional linear/constant warmup.

    Template only — not used by the baseline. Kept here so future work can
    swap in a warmup schedule without adding a new file.
    """

    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_epochs=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers. Got {}".format(
                    milestones
                )
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted, got {}".format(
                    warmup_method
                )
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

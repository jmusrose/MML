"""tool/tools.py

Small utility helpers shared across the DML_v1/CMU pipeline.

Currently only contains ``weight_init`` (mirrored from
``DML_v1/RGB_v1/tool/tools.py``).  Extra helpers can be added here as the
project evolves.
"""

import torch.nn as nn


def weight_init(m):
    """Initialize the weights of a module in-place.

    Intended to be used with ``model.apply(weight_init)``.

    - ``nn.Linear``: Xavier-normal weights, zero bias.
    - ``nn.Conv2d``: Kaiming-normal weights (fan_out, ReLU).
    - ``nn.BatchNorm2d``: weights = 1, bias = 0.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

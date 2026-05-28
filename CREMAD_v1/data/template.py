import torch
import torch.nn.functional as F

config = dict(
    dataset=dict(
        dataset_name='CREMA-D',
        data_root='',
    ),
    visual=dict(
        name='resnet18',
        freeze=False,
        pretrain=False,
        hidden_dim=512,
    ),
    text=dict(
        name='resnet18',
    ),
    loss=dict(
        type='CrossEntropy',
    ),
    head=dict(
        type='MLP',
    ),
    train=dict(
        epoch_dict=100,
        batch_size=64,
        num_workers=8,
        shuffle=True,
        optimizer=dict(
            type='SGD',
            lr=0.01,
            momentum=0.9,
            wc=1e-4,
        ),
        lr_scheduler=dict(
            type='normal',
            patience=70,
        ),
    ),
    test=dict(
        batch_size=64,
        num_workers=8,
    ),
    setting=dict(
        type='CREMA Classification',
        num_class=6,
    ),
    output_dir='.',
    seed=0,
    fps=3,
    use_gpu=True,
    gpu_id=0,
    debug=False,
)

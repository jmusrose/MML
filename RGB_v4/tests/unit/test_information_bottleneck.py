from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


class FakeImageEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        values = torch.arange(
            args.num_image_embeds * args.img_hidden_sz,
            dtype=torch.float32,
        ).view(1, args.num_image_embeds, args.img_hidden_sz)
        self.register_buffer("values", values)

    def forward(self, x):
        return self.values.repeat(x.shape[0], 1, 1)


def make_args(eps_scale=0.0):
    return SimpleNamespace(
        img_hidden_sz=4,
        num_image_embeds=2,
        n_classes=3,
        ib_eps_scale=eps_scale,
    )


def build_classifier(monkeypatch, module_name, eps_scale=0.0):
    module = __import__(module_name, fromlist=["Classifier"])
    monkeypatch.setattr(module, "ImageEncoder", FakeImageEncoder)
    return module.Classifier(make_args(eps_scale=eps_scale))


@pytest.mark.parametrize(
    "module_name",
    ["models.dml_classifier_nyu", "models.dml_classifier_sun"],
)
def test_classifier_returns_information_bottleneck_outputs(monkeypatch, module_name):
    model = build_classifier(monkeypatch, module_name, eps_scale=0.0)
    rgb = torch.zeros(2, 3, 8, 8)
    depth = torch.zeros(2, 3, 8, 8)

    outputs = model(rgb, depth)

    assert len(outputs) == 6
    both_out, rgb_out, depth_out, rgb_latent, depth_latent, ib_losses = outputs
    assert both_out.shape == (2, 3)
    assert rgb_out.shape == (2, 3)
    assert depth_out.shape == (2, 3)
    assert rgb_latent.shape == (2, 3)
    assert depth_latent.shape == (2, 3)
    assert set(ib_losses) == {"rgb", "depth"}
    assert ib_losses["rgb"].ndim == 0
    assert ib_losses["depth"].ndim == 0
    assert ib_losses["rgb"] >= 0
    assert ib_losses["depth"] >= 0


@pytest.mark.parametrize(
    "module_name",
    ["models.dml_classifier_nyu", "models.dml_classifier_sun"],
)
def test_eps_scale_zero_uses_mean_logits_during_training(monkeypatch, module_name):
    model = build_classifier(monkeypatch, module_name, eps_scale=0.0)
    model.train()
    rgb = torch.zeros(2, 3, 8, 8)
    depth = torch.zeros(2, 3, 8, 8)

    _, rgb_out, depth_out, _, _, _ = model(rgb, depth)
    features = torch.flatten(model.rgbenc(rgb), start_dim=1)

    assert torch.allclose(rgb_out, model.rgb_mu(features))
    assert torch.allclose(depth_out, model.depth_mu(features))


@pytest.mark.parametrize(
    "module_name",
    ["models.dml_classifier_nyu", "models.dml_classifier_sun"],
)
def test_eval_mode_uses_mean_logits_without_sampling(monkeypatch, module_name):
    model = build_classifier(monkeypatch, module_name, eps_scale=50.0)
    model.eval()
    rgb = torch.zeros(2, 3, 8, 8)
    depth = torch.zeros(2, 3, 8, 8)

    first = model(rgb, depth)
    second = model(rgb, depth)

    assert torch.allclose(first[1], second[1])
    assert torch.allclose(first[2], second[2])


def test_information_bottleneck_loss_adds_modality_specific_beta_weighted_kl():
    from tool.loss import information_bottleneck_classification_loss

    criterion = nn.CrossEntropyLoss()
    target = torch.tensor([0, 1])
    both_out = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    rgb_out = torch.tensor([[1.5, 0.0], [0.0, 1.5]])
    depth_out = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ib_losses = {"rgb": torch.tensor(4.0), "depth": torch.tensor(2.0)}
    ib_betas = {"rgb": 0.5, "depth": 0.25}

    loss, parts = information_bottleneck_classification_loss(
        criterion,
        both_out,
        rgb_out,
        depth_out,
        target,
        ib_losses,
        ib_beta=0.1,
        ib_betas=ib_betas,
    )

    expected_ce = (
        criterion(both_out, target)
        + criterion(rgb_out, target)
        + criterion(depth_out, target)
    )
    expected_ib = ib_betas["rgb"] * ib_losses["rgb"] + ib_betas["depth"] * ib_losses["depth"]
    assert torch.allclose(loss, expected_ce + expected_ib)
    assert torch.allclose(parts["ib"], expected_ib)
    assert torch.allclose(parts["ib_rgb"], ib_betas["rgb"] * ib_losses["rgb"])
    assert torch.allclose(parts["ib_depth"], ib_betas["depth"] * ib_losses["depth"])

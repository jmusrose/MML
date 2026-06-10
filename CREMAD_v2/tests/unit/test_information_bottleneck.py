#!/usr/bin/env python3
"""Unit tests for CREMAD information bottleneck heads."""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_classifier_source_declares_information_bottleneck_heads():
    source = (PROJECT_ROOT / "model" / "DMLClassifier.py").read_text(encoding="utf-8")

    assert "self.ib_eps_scale" in source
    assert "self.audio_mu" in source
    assert "self.audio_logvar" in source
    assert "self.video_mu" in source
    assert "self.video_logvar" in source
    assert "def _sample_logits" in source
    assert "def _kl_to_standard_normal" in source
    assert "ib_loss" in source


def test_training_source_uses_information_bottleneck_loss():
    source = (PROJECT_ROOT / "DML_cremad.py").read_text(encoding="utf-8")

    assert "information_bottleneck_classification_loss" in source
    assert "ib_beta" in source
    assert "loss_ib" in source


def test_config_source_defines_information_bottleneck_defaults():
    import json

    template = (PROJECT_ROOT / "data" / "template.py").read_text(encoding="utf-8")
    config_text = (PROJECT_ROOT / "data" / "crema.json").read_text(encoding="utf-8")
    config = json.loads(config_text)

    assert "ib_beta" in template
    assert "ib_eps_scale" in template
    assert '"ib_beta"' in config_text
    assert '"ib_eps_scale"' in config_text
    assert config["ib_beta"] == 1e-3
    assert config["ib_eps_scale"] == 0.0


def make_config(eps_scale=0.0):
    return {
        "text": {"name": "resnet18"},
        "visual": {"name": "resnet18"},
        "setting": {"num_class": 6},
        "fps": 3,
        "ib_eps_scale": eps_scale,
    }


def build_classifier(monkeypatch, eps_scale=0.0):
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    import model.DMLClassifier as classifier_module

    class FakeAudioEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            values = torch.arange(512, dtype=torch.float32).view(1, 512)
            self.register_buffer("values", values)

        def forward(self, audio):
            return self.values.repeat(audio.shape[0], 1)

    class FakeVideoEncoder(nn.Module):
        def __init__(self, config, fps=3):
            super().__init__()
            values = torch.arange(512, dtype=torch.float32).view(1, 512)
            self.register_buffer("values", values)

        def forward(self, video):
            return self.values.repeat(video.shape[0], 1)

    monkeypatch.setattr(classifier_module, "AudioEncoder", FakeAudioEncoder)
    monkeypatch.setattr(classifier_module, "VideoEncoder", FakeVideoEncoder)
    return classifier_module.DMLClassifier(make_config(eps_scale=eps_scale))


def test_classifier_returns_information_bottleneck_outputs(monkeypatch):
    torch = pytest.importorskip("torch")
    model = build_classifier(monkeypatch, eps_scale=0.0)
    audio = torch.zeros(2, 1, 8, 8)
    video = torch.zeros(2, 3, 3, 8, 8)

    outputs = model(audio, video)

    assert len(outputs) == 6
    fused, audio_logits, video_logits, audio_latent, video_latent, ib_loss = outputs
    assert fused.shape == (2, 6)
    assert audio_logits.shape == (2, 6)
    assert video_logits.shape == (2, 6)
    assert audio_latent.shape == (2, 6)
    assert video_latent.shape == (2, 6)
    assert torch.allclose(fused, 0.5 * audio_logits + 0.5 * video_logits)
    assert ib_loss.ndim == 0
    assert ib_loss >= 0


def test_eps_scale_zero_uses_mean_logits_during_training(monkeypatch):
    torch = pytest.importorskip("torch")
    model = build_classifier(monkeypatch, eps_scale=0.0)
    model.train()
    audio = torch.zeros(2, 1, 8, 8)
    video = torch.zeros(2, 3, 3, 8, 8)

    _, audio_logits, video_logits, _, _, _ = model(audio, video)
    audio_features = model.audio_encoder(audio)
    video_features = model.video_encoder(video)

    assert torch.allclose(audio_logits, model.audio_mu(audio_features))
    assert torch.allclose(video_logits, model.video_mu(video_features))


def test_eval_mode_uses_mean_logits_without_sampling(monkeypatch):
    torch = pytest.importorskip("torch")
    model = build_classifier(monkeypatch, eps_scale=50.0)
    model.eval()
    audio = torch.zeros(2, 1, 8, 8)
    video = torch.zeros(2, 3, 3, 8, 8)

    first = model(audio, video)
    second = model(audio, video)

    assert torch.allclose(first[1], second[1])
    assert torch.allclose(first[2], second[2])


def test_information_bottleneck_loss_adds_beta_weighted_kl():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    from utils.loss import information_bottleneck_classification_loss

    criterion = nn.CrossEntropyLoss()
    target = torch.tensor([0, 1])
    fused_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    audio_logits = torch.tensor([[1.5, 0.0], [0.0, 1.5]])
    video_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ib_loss = torch.tensor(4.0)
    beta = 0.25

    loss, parts = information_bottleneck_classification_loss(
        criterion,
        fused_logits,
        audio_logits,
        video_logits,
        target,
        ib_loss,
        beta,
    )

    expected_ce = (
        criterion(fused_logits, target)
        + criterion(audio_logits, target)
        + criterion(video_logits, target)
    )
    assert torch.allclose(loss, expected_ce + beta * ib_loss)
    assert torch.allclose(parts["ib"], beta * ib_loss)

#!/usr/bin/env python3
"""Unit tests for RGB_v2-style information bottleneck heads."""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class FakeBertEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, txt, mask, segment):
        values = torch.arange(self.args.hidden_sz, dtype=torch.float32)
        return values.view(1, -1).repeat(txt.shape[0], 1)


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


def test_branch_heads_return_sampled_logits_features_and_ib_loss(
    monkeypatch, mock_args
):
    import models.bert as bert_module
    import models.image as image_module

    monkeypatch.setattr(bert_module, "BertEncoder", FakeBertEncoder)
    monkeypatch.setattr(image_module, "ImageEncoder", FakeImageEncoder)
    mock_args.ib_eps_scale = 0.0

    txt_model = bert_module.BertClf(mock_args)
    img_model = image_module.ImageClf(mock_args)
    txt_model.train()
    img_model.train()

    batch_size = 2
    txt = torch.zeros(batch_size, 4, dtype=torch.long)
    mask = torch.ones(batch_size, 4, dtype=torch.long)
    segment = torch.zeros(batch_size, 4, dtype=torch.long)
    img = torch.zeros(batch_size, 3, 8, 8)

    txt_logits, txt_latent, txt_ib = txt_model(txt, mask, segment)
    img_logits, img_latent, img_ib = img_model(img)

    assert txt_logits.shape == (batch_size, mock_args.n_classes)
    assert img_logits.shape == (batch_size, mock_args.n_classes)
    assert txt_latent.shape == (batch_size, mock_args.hidden_sz)
    assert img_latent.shape == (
        batch_size,
        mock_args.img_hidden_sz * mock_args.num_image_embeds,
    )
    txt_features = txt_model.enc(txt, mask, segment)
    img_features = torch.flatten(img_model.img_encoder(img), start_dim=1)
    assert torch.allclose(txt_latent, txt_model.mu(txt_features))
    assert torch.allclose(img_latent, img_model.mu(img_features))
    assert torch.allclose(txt_logits, txt_model.clf(txt_latent))
    assert torch.allclose(img_logits, img_model.clf(img_latent))
    assert txt_ib.ndim == 0
    assert img_ib.ndim == 0
    assert txt_ib >= 0
    assert img_ib >= 0


def test_eval_mode_uses_mean_logits_without_sampling(monkeypatch, mock_args):
    import models.bert as bert_module

    monkeypatch.setattr(bert_module, "BertEncoder", FakeBertEncoder)
    mock_args.ib_eps_scale = 50.0
    model = bert_module.BertClf(mock_args)
    model.eval()

    txt = torch.zeros(2, 4, dtype=torch.long)
    mask = torch.ones(2, 4, dtype=torch.long)
    segment = torch.zeros(2, 4, dtype=torch.long)

    first_logits, first_latent, _ = model(txt, mask, segment)
    second_logits, second_latent, _ = model(txt, mask, segment)

    assert torch.allclose(first_logits, second_logits)
    assert torch.allclose(first_latent, second_latent)
    assert torch.allclose(first_logits, model.clf(first_latent))


def test_classifier_returns_rgb_v2_style_information_bottleneck_outputs(
    monkeypatch, mock_args
):
    import models.dml_classifier as classifier_module

    class FakeTextClf(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.enc = nn.Linear(1, 1)
            self.args = args

        def forward(self, txt, mask, segment):
            logits = torch.ones(txt.shape[0], self.args.n_classes)
            latent = torch.full((txt.shape[0], self.args.hidden_sz), 2.0)
            return logits, latent, torch.tensor(1.25)

    class FakeImageClf(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.img_encoder = nn.Linear(1, 1)
            self.args = args

        def forward(self, img):
            logits = torch.full((img.shape[0], self.args.n_classes), 3.0)
            latent = torch.full(
                (
                    img.shape[0],
                    self.args.img_hidden_sz * self.args.num_image_embeds,
                ),
                4.0,
            )
            return logits, latent, torch.tensor(2.75)

    monkeypatch.setattr(classifier_module, "BertClf", FakeTextClf)
    monkeypatch.setattr(classifier_module, "ImageClf", FakeImageClf)
    model = classifier_module.Classifier(mock_args)

    txt = torch.zeros(2, 4, dtype=torch.long)
    mask = torch.ones(2, 4, dtype=torch.long)
    segment = torch.zeros(2, 4, dtype=torch.long)
    img = torch.zeros(2, 3, 8, 8)

    outputs = model(txt, mask, segment, img)

    assert len(outputs) == 6
    fused, txt_logits, img_logits, txt_latent, img_latent, ib_loss = outputs
    assert torch.allclose(fused, 0.5 * txt_logits + 0.5 * img_logits)
    assert txt_latent.shape == (2, mock_args.hidden_sz)
    assert img_latent.shape == (
        2,
        mock_args.img_hidden_sz * mock_args.num_image_embeds,
    )
    assert torch.allclose(txt_latent, torch.full_like(txt_latent, 2.0))
    assert torch.allclose(img_latent, torch.full_like(img_latent, 4.0))
    assert torch.allclose(ib_loss, torch.tensor(4.0))


def test_information_bottleneck_loss_adds_beta_weighted_kl():
    from utils.loss import information_bottleneck_classification_loss

    criterion = nn.CrossEntropyLoss()
    target = torch.tensor([0, 1])
    fused_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    txt_logits = torch.tensor([[1.5, 0.0], [0.0, 1.5]])
    img_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ib_loss = torch.tensor(4.0)
    beta = 0.25

    loss, parts = information_bottleneck_classification_loss(
        criterion,
        fused_logits,
        txt_logits,
        img_logits,
        target,
        ib_loss,
        beta,
    )

    expected_ce = (
        criterion(fused_logits, target)
        + criterion(txt_logits, target)
        + criterion(img_logits, target)
    )
    assert torch.allclose(loss, expected_ce + beta * ib_loss)
    assert torch.allclose(parts["ib"], beta * ib_loss)

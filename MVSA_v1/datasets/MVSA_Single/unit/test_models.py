#!/usr/bin/env python3
"""Unit tests for model components: image encoder shapes, fusion arithmetic."""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.image import ImageEncoder, ImageClf
from models.dml_classifier import Classifier


class TestImageEncoder:
    """Tests for ImageEncoder output shapes."""

    def test_output_shape(self, mock_args):
        encoder = ImageEncoder(mock_args)
        encoder.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = encoder(x)
        # Expected: B x num_image_embeds x img_hidden_sz
        assert out.shape == (2, mock_args.num_image_embeds, mock_args.img_hidden_sz)

    def test_single_embed(self, mock_args):
        mock_args.num_image_embeds = 1
        encoder = ImageEncoder(mock_args)
        encoder.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (1, 1, 2048)


class TestImageClf:
    """Tests for ImageClf."""

    def test_output_shapes(self, mock_args):
        clf = ImageClf(mock_args)
        clf.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits, features = clf(x)
        assert logits.shape == (2, mock_args.n_classes)
        assert features.shape == (
            2,
            mock_args.img_hidden_sz * mock_args.num_image_embeds,
        )


class TestClassifier:
    """Tests for the full DML Classifier."""

    def test_forward_output_structure(self, mock_args):
        """Classifier forward should return 5-tuple with correct shapes."""
        model = Classifier(mock_args)
        model.eval()

        B = 2
        seq_len = 10
        txt = torch.randint(0, 1000, (B, seq_len))
        mask = torch.ones(B, seq_len).long()
        segment = torch.zeros(B, seq_len).long()
        img = torch.randn(B, 3, 224, 224)

        with torch.no_grad():
            fused, txt_logits, img_logits, txt_feat, img_feat = model(
                txt, mask, segment, img
            )

        assert fused.shape == (B, mock_args.n_classes)
        assert txt_logits.shape == (B, mock_args.n_classes)
        assert img_logits.shape == (B, mock_args.n_classes)
        assert txt_feat.shape == (B, mock_args.hidden_sz)
        assert img_feat.shape == (
            B,
            mock_args.img_hidden_sz * mock_args.num_image_embeds,
        )

    def test_fusion_is_average(self, mock_args):
        """Fused logits should be exactly 0.5 * txt + 0.5 * img."""
        model = Classifier(mock_args)
        model.eval()

        B = 2
        seq_len = 10
        txt = torch.randint(0, 1000, (B, seq_len))
        mask = torch.ones(B, seq_len).long()
        segment = torch.zeros(B, seq_len).long()
        img = torch.randn(B, 3, 224, 224)

        with torch.no_grad():
            fused, txt_logits, img_logits, _, _ = model(txt, mask, segment, img)

        expected = 0.5 * txt_logits + 0.5 * img_logits
        assert torch.allclose(fused, expected, atol=1e-6)

    def test_no_cross_modal_interaction(self, mock_args):
        """Text and image branches should be independent."""
        model = Classifier(mock_args)
        model.eval()

        B = 1
        seq_len = 10
        txt = torch.randint(0, 1000, (B, seq_len))
        mask = torch.ones(B, seq_len).long()
        segment = torch.zeros(B, seq_len).long()
        img1 = torch.randn(B, 3, 224, 224)
        img2 = torch.randn(B, 3, 224, 224)

        with torch.no_grad():
            _, txt_logits1, _, _, _ = model(txt, mask, segment, img1)
            _, txt_logits2, _, _, _ = model(txt, mask, segment, img2)

        # Text logits should be the same regardless of image input
        assert torch.allclose(txt_logits1, txt_logits2, atol=1e-6)

"""Property tests for checkpoint save / load / forward round-trip.

Feature: dml-cmu-multimodal
Property 17: 检查点保存-加载-前向往返一致性
Validates: Requirements 8.5, 11.1, 11.2

The property under test is: for any ``Classifier`` state, persisting via
``torch.save`` and restoring via ``torch.load`` + ``load_state_dict`` must
reproduce the original ``forward`` output bit-for-bit (within float
equality tolerance) on identical inputs.

To keep the test cheap we monkey-patch ``transformers.BertModel`` with a
lightweight ``nn.Embedding + nn.Linear`` stand-in (preserving the
``last_hidden_state`` contract used by ``TextEncoder``); the round-trip
property is independent of the BERT internals.
"""
import io
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from models.dml_classifier_mosi import Classifier


# Tiny dimensions so the test runs in milliseconds even at max_examples=100.
_VOCAB = 200
_TEXT_HIDDEN = 16
_VISION_DIM = 8
_AUDIO_DIM = 10
_HIDDEN = 12
_T = 6
_L = 5


class _FakeBertOutput:
    """Object exposing ``last_hidden_state`` to mirror HF BERT's ModelOutput."""

    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeBertModel(nn.Module):
    """Lightweight stand-in for ``transformers.BertModel``.

    Replicates the ``(input_ids, attention_mask) -> obj.last_hidden_state``
    contract used by :class:`models.text.TextEncoder`, with real parameters
    so they participate in the checkpoint round-trip.
    """

    @classmethod
    def from_pretrained(cls, name, *args, **kwargs):
        return cls()

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(_VOCAB, _TEXT_HIDDEN)
        self.lin = nn.Linear(_TEXT_HIDDEN, _TEXT_HIDDEN)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        x = self.lin(x)
        return _FakeBertOutput(x)


@pytest.fixture(autouse=True)
def _patch_bert(monkeypatch):
    """Replace ``transformers.BertModel`` with the tiny fake for every test."""
    import transformers

    monkeypatch.setattr(transformers, "BertModel", _FakeBertModel)


def _make_args(pool_strategy: str) -> SimpleNamespace:
    return SimpleNamespace(
        vision_dim=_VISION_DIM,
        audio_dim=_AUDIO_DIM,
        hidden_sz=_HIDDEN,
        num_heads=2,
        num_layers=1,
        conv_kernel_size=3,
        dropout=0.0,
        text_hidden_sz=_TEXT_HIDDEN,
        n_classes=1,
        bert_model_name="bert-base-uncased",
        freeze_bert=False,
        pool_strategy=pool_strategy,
    )


def _make_inputs(batch_size: int, input_seed: int):
    """Build a deterministic input batch for a given ``input_seed``."""
    g = torch.Generator().manual_seed(input_seed)
    vision = torch.randn(batch_size, _T, _VISION_DIM, generator=g)
    audio = torch.randn(batch_size, _T, _AUDIO_DIM, generator=g)
    vision_mask = torch.zeros(batch_size, _T, dtype=torch.bool)
    audio_mask = torch.zeros(batch_size, _T, dtype=torch.bool)
    # Mark the trailing position of the first sample as padding so the
    # 'default' pool branch with a real masked position is exercised too.
    vision_mask[0, -1] = True
    audio_mask[0, -1] = True
    text_input_ids = torch.randint(0, _VOCAB, (batch_size, _L), generator=g)
    text_attention_mask = torch.ones(batch_size, _L, dtype=torch.long)
    return (
        vision,
        vision_mask,
        audio,
        audio_mask,
        text_input_ids,
        text_attention_mask,
    )


@given(
    init_seed=st.integers(min_value=0, max_value=2**16 - 1),
    input_seed=st.integers(min_value=0, max_value=2**16 - 1),
    batch_size=st.integers(min_value=1, max_value=3),
    pool_strategy=st.sampled_from(["last", "default"]),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_checkpoint_roundtrip_forward_consistency(
    init_seed, input_seed, batch_size, pool_strategy
):
    """Feature: dml-cmu-multimodal, Property 17: 检查点保存-加载-前向往返一致性.

    For any ``Classifier`` weights ``W`` and any input batch ``x``,
    ``forward_W(x) == forward_(load(save(W)))(x)`` within float tolerance.
    """
    args = _make_args(pool_strategy=pool_strategy)

    # Build model A with a deterministic init.
    torch.manual_seed(init_seed)
    model_a = Classifier(args).eval()

    inputs = _make_inputs(batch_size, input_seed)
    with torch.no_grad():
        out_a = model_a(*inputs)

    # Persist using the project's checkpoint schema (Requirement 11.1).
    buf = io.BytesIO()
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model_a.state_dict(),
            "optimizer_state_dict": {},
            "metrics": {
                "mae": 0.0, "corr": 0.0, "acc7": 0.0, "acc2": 0.0, "f1": 0.0
            },
            "clean_acc": 0.0,
        },
        buf,
    )
    buf.seek(0)

    # Build model B with a *different* init then load A's weights into it.
    torch.manual_seed(init_seed + 1)
    model_b = Classifier(args).eval()
    state = torch.load(buf, map_location="cpu")
    model_b.load_state_dict(state["model_state_dict"])

    with torch.no_grad():
        out_b = model_b(*inputs)

    assert len(out_a) == len(out_b) == 7
    for i, (a, b) in enumerate(zip(out_a, out_b)):
        assert a.shape == b.shape, (
            f"output[{i}] shape changed after round-trip: {a.shape} -> {b.shape}"
        )
        max_diff = (a - b).abs().max().item()
        assert torch.allclose(a, b, atol=1e-6, rtol=0), (
            f"output[{i}] differs after checkpoint round-trip "
            f"(max abs diff = {max_diff})"
        )

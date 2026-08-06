import pytest
import torch
import torch.nn as nn

from brainmarks.classifiers import ClassifierGrid, create_classifier, masked_mean
from brainmarks.models.base import Embeddings, as_embeddings

CLASSIFIER_NAMES = ["attn", "linear", "mlp"]


def test_masked_mean_ignores_padding():
    x = torch.randn(2, 5, 4)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, 3:] = True  # last two positions padded
    got = masked_mean(x, mask)
    want = x[:, :3].mean(dim=1)
    assert torch.allclose(got, want)


def test_masked_mean_none_is_plain_mean():
    x = torch.randn(2, 5, 4)
    assert torch.allclose(masked_mean(x, None), x.mean(dim=1))


def test_masked_mean_fully_padded_row_is_zero():
    x = torch.randn(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    assert torch.allclose(masked_mean(x, mask), torch.zeros(1, 4))


@pytest.mark.parametrize("mask_padding", [False, True])
def test_masked_mean_accumulates_low_precision_inputs_in_float32(mask_padding):
    x = torch.full((1, 4096, 4), 32.0, dtype=torch.float16)
    mask = None
    if mask_padding:
        mask = torch.zeros(1, 4096, dtype=torch.bool)
        mask[:, -16:] = True
        x[:, -16:] = 0

    got = masked_mean(x, mask)

    assert got.dtype == torch.float32
    assert torch.isfinite(got).all()
    assert torch.equal(got, torch.full((1, 4), 32.0))


def _padded_inputs(seed=0, B=2, N=6, D=128, M=4):
    torch.manual_seed(seed)
    x = torch.randn(B, N, D)
    # large-magnitude garbage that must be ignored when masked
    garbage = 100.0 * torch.randn(B, M, D)
    x_pad = torch.cat([x, garbage], dim=1)
    mask = torch.zeros(B, N + M, dtype=torch.bool)
    mask[:, N:] = True
    return x, x_pad, mask


@pytest.mark.parametrize("name", CLASSIFIER_NAMES)
def test_classifier_invariant_to_masked_padding(name):
    """Masking padded positions reproduces the unpadded result exactly."""
    x, x_pad, mask = _padded_inputs()
    clf = create_classifier(name, in_dim=x.shape[-1], out_dim=3).eval()
    with torch.no_grad():
        ref = clf(x)
        out = clf(x_pad, mask=mask)
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("name", CLASSIFIER_NAMES)
def test_unmasked_padding_leaks_into_output(name):
    """Sanity: without the mask, padding does change the output (so the mask matters)."""
    x, x_pad, _ = _padded_inputs()
    clf = create_classifier(name, in_dim=x.shape[-1], out_dim=3).eval()
    with torch.no_grad():
        ref = clf(x)
        leaked = clf(x_pad)
    assert not torch.allclose(ref, leaked, atol=1e-4)


def test_classifier_grid_threads_patch_mask():
    """ClassifierGrid forwards patch_mask from the backbone into the head."""
    x, x_pad, mask = _padded_inputs()

    class PadBackbone(nn.Module):
        def forward(self, batch):
            return Embeddings(None, None, x_pad, mask)

    class CleanBackbone(nn.Module):
        def forward(self, batch):
            return Embeddings(None, None, x, None)

    clf = create_classifier("attn", in_dim=x.shape[-1], out_dim=3).eval()
    grid_pad = ClassifierGrid(PadBackbone(), "patch", {(0.0, 0.0): clf})
    grid_clean = ClassifierGrid(CleanBackbone(), "patch", {(0.0, 0.0): clf})
    with torch.no_grad():
        out_pad = grid_pad({})
        out_clean = grid_clean({})
    assert torch.allclose(out_pad, out_clean, atol=1e-4, rtol=1e-4)


def test_as_embeddings_normalizes_bare_tuple():
    bare = as_embeddings((None, None, torch.zeros(1, 2, 3)))
    assert isinstance(bare, Embeddings)
    assert bare.patch_mask is None

    tup = as_embeddings(Embeddings(None, None, None, torch.ones(1, 2, dtype=torch.bool)))
    assert isinstance(tup, Embeddings)
    assert tup.patch_mask is not None

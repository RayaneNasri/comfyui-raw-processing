import torch

from algorithms.lens_correction._chromatic_aberration import (
    correct_chromatic_aberration,
)
from algorithms.lens_correction._distortion import correct_distortion
from algorithms.lens_correction._vignetting import correct_vignetting


# Vignetting


def test_vignetting_identity():
    img = torch.rand(32, 32, 3)
    out = correct_vignetting(img, 0.0, 0.0)
    assert torch.equal(out, img)


def test_vignetting_shape_preserved():
    img = torch.rand(64, 48, 3)
    out = correct_vignetting(img, 0.5, 0.1)
    assert out.shape == img.shape


def test_vignetting_brightens_corners():
    img = torch.full((64, 64, 3), 0.4)
    out = correct_vignetting(img, 0.5, 0.0)
    center = out[32, 32, 0].item()
    corner = out[0, 0, 0].item()
    assert corner > center


def test_vignetting_clamped():
    img = torch.ones(32, 32, 3)
    out = correct_vignetting(img, 2.0, 2.0)
    assert out.max().item() <= 1.0
    assert out.min().item() >= 0.0


def test_vignetting_gain_map_overrides_params():
    img = torch.full((8, 8, 3), 0.5)
    gain_map = torch.full((8, 8, 3), 2.0)
    out = correct_vignetting(img, 0.0, 0.0, gain_map=gain_map)
    assert torch.allclose(out, torch.ones(8, 8, 3))


def test_vignetting_gain_map_shape_preserved():
    img = torch.rand(16, 24, 3)
    gain_map = torch.ones(16, 24, 3) * 1.2
    out = correct_vignetting(img, 0.0, 0.0, gain_map=gain_map)
    assert out.shape == img.shape


# Distortion


def test_distortion_identity():
    img = torch.rand(64, 64, 3)
    out = correct_distortion(img, 0.0, 0.0)
    assert torch.equal(out, img)


def test_distortion_shape_preserved():
    img = torch.rand(64, 48, 3)
    out = correct_distortion(img, -0.1, 0.0)
    assert out.shape == img.shape


def test_distortion_non_square_shape_preserved():
    img = torch.rand(100, 150, 3)
    out = correct_distortion(img, 0.05, -0.01)
    assert out.shape == img.shape


def test_distortion_clamped():
    img = torch.rand(32, 32, 3)
    out = correct_distortion(img, 0.3, 0.0)
    assert out.max().item() <= 1.0
    assert out.min().item() >= 0.0


# Chromatic Aberration


def test_ca_identity():
    img = torch.rand(32, 32, 3)
    out = correct_chromatic_aberration(img, 1.0, 1.0)
    assert torch.allclose(out, img)


def test_ca_shape_preserved():
    img = torch.rand(64, 48, 3)
    out = correct_chromatic_aberration(img, 1.01, 0.99)
    assert out.shape == img.shape


def test_ca_g_channel_unchanged():
    img = torch.rand(32, 32, 3)
    out = correct_chromatic_aberration(img, 1.02, 0.98)
    assert torch.allclose(out[..., 1], img[..., 1])


def test_ca_clamped():
    img = torch.rand(32, 32, 3)
    out = correct_chromatic_aberration(img, 1.05, 0.95)
    assert out.max().item() <= 1.0
    assert out.min().item() >= 0.0


def test_ca_only_red_scale():
    img = torch.rand(32, 32, 3)
    out = correct_chromatic_aberration(img, 1.02, 1.0)
    # B channel must be unchanged when blue_scale=1.0
    assert torch.allclose(out[..., 2], img[..., 2])


def test_ca_only_blue_scale():
    img = torch.rand(32, 32, 3)
    out = correct_chromatic_aberration(img, 1.0, 0.98)
    # R channel must be unchanged when red_scale=1.0
    assert torch.allclose(out[..., 0], img[..., 0])

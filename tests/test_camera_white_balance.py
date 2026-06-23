import torch
import pytest
from torch.testing import assert_close

from algorithms.white_balance import camera_white_balance, raw_wb_gains_to_rgb


def test_raw_wb_gains_to_rgb_from_len3_is_identity():
    gains = torch.tensor([2.0, 1.0, 0.5])
    out = raw_wb_gains_to_rgb(gains)
    assert_close(out, gains)


def test_raw_wb_gains_to_rgb_merges_two_green_values():
    gains = torch.tensor([2.0, 1.0, 0.5, 3.0])
    out = raw_wb_gains_to_rgb(gains)
    expected = torch.tensor([2.0, 2.0, 0.5])
    assert_close(out, expected)


def test_raw_wb_gains_to_rgb_raises_for_too_short_input():
    with pytest.raises(ValueError, match="at least 3"):
        raw_wb_gains_to_rgb(torch.tensor([1.0, 2.0]))


def test_preserves_shape():
    img = torch.rand(32, 48, 3)
    wb_gains = torch.tensor([2.0, 1.0, 1.5, 1.0])
    out = camera_white_balance(img, wb_gains)
    assert out.shape == img.shape


def test_merges_both_green_gains_from_raw_metadata():
    img = torch.ones(4, 4, 3) * 0.5

    wb_gains_a = torch.tensor([2.0, 1.0, 1.5, 3.0])
    wb_gains_b = torch.tensor([2.0, 2.0, 1.5, 2.0])

    out_a = camera_white_balance(img, wb_gains_a)
    out_b = camera_white_balance(img, wb_gains_b)

    assert_close(out_a, out_b)


def test_uses_other_green_gain_if_one_is_zero():
    img = torch.ones(8, 8, 3) * 0.5
    wb_gains = torch.tensor([2.0, 0.0, 1.0, 2.0])

    out = camera_white_balance(img, wb_gains)

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_strength_slider_zero_keeps_original_image():
    img = torch.rand(16, 16, 3)
    wb_gains = torch.tensor([2.0, 1.0, 0.5, 1.0])
    out = camera_white_balance(img, wb_gains, strength=0.0)
    assert_close(out, img)


def test_strength_slider_changes_image_when_nonzero():
    img = torch.ones(8, 8, 3) * 0.4
    wb_gains = torch.tensor([2.0, 1.0, 0.5, 1.0])

    out_0 = camera_white_balance(img, wb_gains, strength=0.0)
    out_1 = camera_white_balance(img, wb_gains, strength=1.0)

    assert not torch.allclose(out_0, out_1)

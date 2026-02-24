import torch
import pytest 

import torch
import pytest

from algorithms.exposure_compensation._exposure_compensation import exposure_compensation


def test_ev_zero_returns_same_image():
    img = torch.rand(3, 4, 4)
    out = exposure_compensation(img, 0.0)
    assert torch.allclose(out, img)


def test_positive_ev_doubles_values():
    img = torch.ones(3, 4, 4)
    out = exposure_compensation(img, 1.0)
    expected = img * 2.0
    assert torch.allclose(out, expected)


def test_negative_ev_halves_values():
    img = torch.ones(3, 4, 4)
    out = exposure_compensation(img, -1.0)
    expected = img * 0.5
    assert torch.allclose(out, expected)


def test_batch_dimension():
    img = torch.ones(2, 3, 8, 8)
    out = exposure_compensation(img, 2.0)
    expected = img * 4.0
    assert torch.allclose(out, expected)


def test_dtype_preserved():
    img = torch.rand(3, 4, 4, dtype=torch.float32)
    out = exposure_compensation(img, 1.0)
    assert out.dtype == img.dtype
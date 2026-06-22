import torch
import pytest

from algorithms.gc.iec_gamma_correction import iec_gamma_correction


def test_iec_gamma_correction_known_values():
    """Tests the two branches of the sRGB piecewise formula with predictable values."""
    img = torch.tensor([[[0.0, 0.0031308, 1.0]]])

    out = iec_gamma_correction(img)

    expected = torch.tensor([[[0.0, 0.04045, 1.0]]])
    assert torch.allclose(out, expected, atol=1e-5), (
        "Known sRGB values do not match expected output."
    )


def test_iec_gamma_correction_linear_branch():
    """Values <= 0.0031308 must use the linear branch: out = 12.92 * x."""
    img = torch.tensor([[[0.001, 0.002, 0.0031308]]])

    out = iec_gamma_correction(img)

    expected = 12.92 * img
    assert torch.allclose(out, expected, atol=1e-5), (
        "Linear branch (12.92 * x) was not applied correctly for small values."
    )


def test_iec_gamma_correction_power_branch():
    """Values > 0.0031308 must use the power branch: out = 1.055 * x^(1/2.4) - 0.055."""
    img = torch.tensor([[[0.1, 0.5, 0.9]]])

    out = iec_gamma_correction(img)

    expected = 1.055 * torch.pow(img, 1 / 2.4) - 0.055
    assert torch.allclose(out, expected, atol=1e-5), (
        "Power branch (1.055 * x^(1/2.4) - 0.055) was not applied correctly."
    )


def test_iec_gamma_correction_black_and_white():
    """Pure black and pure white must map exactly to 0.0 and 1.0."""
    img = torch.tensor([[[0.0, 1.0]]])

    out = iec_gamma_correction(img)

    assert out[0, 0, 0].item() == pytest.approx(0.0, abs=1e-6), (
        "Pure black should remain 0.0."
    )
    assert out[0, 0, 1].item() == pytest.approx(1.0, abs=1e-6), (
        "Pure white should remain 1.0."
    )


def test_iec_gamma_correction_clamps_above_one():
    """Input values > 1.0 must be clamped before processing; output must not exceed 1.0."""
    img = torch.tensor([[[1.5, 2.0, 10.0]]])

    out = iec_gamma_correction(img)

    assert torch.all(out <= 1.0), (
        "Output contains values above 1.0 for saturated inputs."
    )


def test_iec_gamma_correction_clamps_below_zero():
    """Negative input values must be clamped; output must not go below 0.0."""
    img = torch.tensor([[[-0.5, -1.0, -100.0]]])

    out = iec_gamma_correction(img)

    assert torch.all(out >= 0.0), "Output contains negative values for negative inputs."


def test_iec_gamma_correction_output_range():
    """For any valid input in [0, 1], the output must stay within [0, 1]."""
    img = torch.rand((100, 100, 3))

    out = iec_gamma_correction(img)

    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), (
        "Output is out of the [0, 1] range for a valid random input."
    )


def test_iec_gamma_correction_preserves_shape():
    """The output tensor must have the exact same shape as the input tensor."""
    shape = (1080, 1920, 3)
    img = torch.rand(shape)

    out = iec_gamma_correction(img)

    assert out.shape == shape, f"Shape changed! Expected {shape}, got {out.shape}."


def test_iec_gamma_correction_monotonically_increasing():
    """A brighter input must always produce a brighter (or equal) output."""
    img = torch.linspace(0.0, 1.0, steps=1000).unsqueeze(0).unsqueeze(-1)

    out = iec_gamma_correction(img)
    diffs = out[0, 1:, 0] - out[0, :-1, 0]

    assert torch.all(diffs >= -1e-7), (
        "Output is not monotonically increasing with respect to input."
    )


def test_iec_gamma_correction_rejects_wrong_type():
    """The function should fail loudly if given a list instead of a Tensor."""
    img_list = [[[0.5, 0.5, 0.5]]]

    with pytest.raises((TypeError, AttributeError)):  # type: ignore
        iec_gamma_correction(img_list)  # type: ignore

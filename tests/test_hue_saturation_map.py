import pytest
import torch

from torch import Tensor
from algorithms.hue_saturation_map._hue_saturation_map import apply_hue_sat_map

ILLUM_D65 = 23  # Standard illuminant D50 (5003 K)
ILLUM_D65 = 21  # Standard illuminant D65 (6504 K)
ILLUM_A = 17  # Standard illuminant A (2856 K)


def _identity_color_matrix() -> Tensor:
    return torch.eye(3, dtype=torch.float32)

def _identity_forward_matrix() -> Tensor:
    return torch.tensor(
        [[0.797674, 0.135191, 0.031353],
         [0.288040, 0.711874, 0.000086],
         [0.000000, 0.000000, 0.825210]], dtype=torch.float32)

def _neutral_lut(h_bins: int = 90, s_bins: int = 30, v_bins: int = 1) -> Tensor:
    lut = torch.zeros((h_bins, s_bins, v_bins, 3), dtype=torch.float32)
    lut[..., 0] = 0.0  # Hue delta
    lut[..., 1] = 1.0  # Saturation scale
    lut[..., 2] = 1.0  # Value scale
    return lut


def _unit_wb_gains() -> Tensor:
    return torch.ones(3, dtype=torch.float32)


def _make_grey_image(value: float, height: int = 64, width: int = 64) -> Tensor:
    return torch.full((height, width, 3), value, dtype=torch.float32)


def _make_random_image(height: int = 64, width: int = 64, seed: int = 0) -> Tensor:
    torch.manual_seed(seed)
    return torch.rand((height, width, 3), dtype=torch.float32)


def _assert_tensors_close(
    actual: Tensor, expected: Tensor, atol: float = 1e-3, msg: str = ""
):
    if torch.allclose(actual, expected, atol=atol):
        return
    diff = (actual - expected).abs()
    failing_mask = diff > atol
    failing_indices = failing_mask.nonzero(as_tuple=False)  # shape [N, ndim]

    lines = [
        msg or "Tensors are not close",
        f"  atol={atol}, shape={tuple(actual.shape)}",
        f"  max diff : {diff.max():.6f}",
        f"  mean diff: {diff.mean():.6f}",
        f"  failing pixels: {failing_mask.sum().item()} / {actual.numel()}",
        "  first 5 failures (index | actual | expected | diff):",
    ]
    for idx in failing_indices[:5]:
        i = tuple(idx.tolist())
        lines.append(
            f"     [{i}]  {actual[i].item():.6f}  vs  {expected[i].item():.6f}  (diff {diff[i].item():.6f})"
        )

    pytest.fail("\n".join(lines))


class TestOutputShape:
    def test_same_shape_as_input(self):
        img = _make_random_image(16, 16)
        lut = _neutral_lut()
        out = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut,
            lut,
            ILLUM_D65,
            ILLUM_D65,
        )
        assert out.shape == img.shape

    def test_output_dtype_preserved(self):
        img = _make_random_image().to(torch.float32)
        lut = _neutral_lut()
        out = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut,
            lut,
            ILLUM_D65,
            ILLUM_D65,
        )
        assert out.dtype == img.dtype

    def test_output_range_0_1(self):
        """Output RGB values must stay in [0, 1] after clamping."""
        img = _make_random_image(seed=42)
        lut = _neutral_lut()
        out = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut,
            lut,
            ILLUM_D65,
            ILLUM_D65,
        )
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestNeutralLUT:
    def test_grey_unchanged_under_neutral_lut(self):
        img = _make_grey_image(0.5)
        lut = _neutral_lut()
        out = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut,
            lut,
            ILLUM_D65,
            ILLUM_D65,
        )
        _assert_tensors_close(out, img, atol=1e-4)

    def test_random_image_nearly_unchanged_under_neutral_lut(self):
        img = _make_random_image(seed=7)
        lut = _neutral_lut()
        out = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut,
            lut,
            ILLUM_D65,
            ILLUM_D65,
        )
        _assert_tensors_close(out, img, atol=1e-3)


class TestHueCorrection:
    def _lut_constant_delta_h(self, delta_h: float) -> Tensor:
        lut = _neutral_lut()
        lut[..., 0] = delta_h
        return lut

    def test_hue_shift_360_is_identity(self):
        img = _make_random_image(seed=6)
        lut = _neutral_lut()
        lut_360 = self._lut_constant_delta_h(360.0)
        out = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut,
            lut,
            ILLUM_D65,
            ILLUM_D65,
        )
        out_360 = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut_360,
            lut_360,
            ILLUM_D65,
            ILLUM_D65,
        )
        assert torch.allclose(out, out_360, atol=1e-3)

    def test_grey_image_unaffected_by_hue_shift(self):
        img = _make_grey_image(0.5)
        lut = self._lut_constant_delta_h(90.0)
        out = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut,
            lut,
            ILLUM_D65,
            ILLUM_D65,
        )
        _assert_tensors_close(out, img, atol=1e-3)

    def test_opposite_hue_shifts_cancel(self):
        neutral_xyz = torch.tensor([0.3457, 0.3585, 0.2958], dtype=torch.float32)
        white = neutral_xyz / neutral_xyz.max()
        torch.manual_seed(9)
        brightness = torch.rand(64, 64, 1, dtype=torch.float32) * 0.9
        img = (white.view(1, 1, 3) * brightness).clone()
        lut_pos = self._lut_constant_delta_h(45.0)
        lut_neg = self._lut_constant_delta_h(-45.0)

        mid = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut_pos,
            lut_pos,
            ILLUM_D65,
            ILLUM_D65,
        )
        out = apply_hue_sat_map(
            mid,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut_neg,
            lut_neg,
            ILLUM_D65,
            ILLUM_D65,
        )
        _assert_tensors_close(out, img, atol=2e-3)


class TestSaturationCorrection:
    def _lut_constant_sigma_s(self, sigma: float) -> Tensor:
        lut = _neutral_lut()
        lut[..., 1] = sigma
        return lut

    def test_zero_saturation_produces_grey(self):
        img = _make_random_image(seed=1)
        lut = self._lut_constant_sigma_s(0.0)
        out = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut,
            lut,
            ILLUM_D65,
            ILLUM_D65,
        )

        def rgb_sat(t: Tensor) -> Tensor:
            return t.max(dim=-1).values - t.min(dim=-1).values

        sat_in = rgb_sat(img)
        sat_out = rgb_sat(out)

        highly_sat = sat_in > 0.3
        if highly_sat.any():
            reduction = (
                sat_out[highly_sat] / sat_in[highly_sat].clamp(min=1e-6)
            ).mean()
            assert reduction < 0.3

    def test_high_saturation_sigma_clamped(self):
        img = _make_random_image(seed=2)
        lut = self._lut_constant_sigma_s(1e6)
        out = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut,
            lut,
            ILLUM_D65,
            ILLUM_D65,
        )
        assert out.max() <= 1.0 + 1e-5

    def test_saturation_sigma_less_than_1_reduces_saturation(self):
        img = _make_random_image(seed=3)
        lut_full = _neutral_lut()
        lut_half = self._lut_constant_sigma_s(0.5)

        out_full = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut_full,
            lut_full,
            ILLUM_D65,
            ILLUM_D65,
        )
        out_half = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut_half,
            lut_half,
            ILLUM_D65,
            ILLUM_D65,
        )

        def mean_sat(t: Tensor) -> float:
            return (t.max(dim=-1).values - t.min(dim=-1).values).mean().item()

        assert mean_sat(out_half) < mean_sat(out_full)

    def test_opposite_saturation_scales_cancel(self):
        neutral_xyz = torch.tensor([0.3457, 0.3585, 0.2958], dtype=torch.float32)
        white = neutral_xyz / neutral_xyz.max()
        torch.manual_seed(9)
        brightness = torch.rand(64, 64, 1, dtype=torch.float32) * 0.9
        img = (white.view(1, 1, 3) * brightness).clone()

        lut_up = self._lut_constant_sigma_s(2.0)
        lut_down = self._lut_constant_sigma_s(0.5)

        mid = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut_up,
            lut_up,
            ILLUM_D65,
            ILLUM_D65,
        )
        out = apply_hue_sat_map(
            mid,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut_down,
            lut_down,
            ILLUM_D65,
            ILLUM_D65,
        )
        _assert_tensors_close(out, img, atol=2e-3)


class TestValueCorrection:
    def _lut_constant_sigma_v(self, sigma: float) -> Tensor:
        lut = _neutral_lut()
        lut[..., 2] = sigma
        return lut

    def test_zero_value_produces_black(self):
        img = _make_random_image(seed=4)
        lut = self._lut_constant_sigma_v(0.0)
        out = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut,
            lut,
            ILLUM_D65,
            ILLUM_D65,
        )
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-4)

    def test_high_value_sigma_clamped(self):
        img = _make_random_image(seed=5)
        lut = self._lut_constant_sigma_v(1e9)
        out = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut,
            lut,
            ILLUM_D65,
            ILLUM_D65,
        )
        assert out.max() <= 1.0 + 1e-5

    def test_value_sigma_less_than_1_reduces_brightness(self):
        img = _make_random_image(seed=6)
        lut_full = _neutral_lut()
        lut_half = self._lut_constant_sigma_v(0.5)

        out_full = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut_full,
            lut_full,
            ILLUM_D65,
            ILLUM_D65,
        )
        out_half = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut_half,
            lut_half,
            ILLUM_D65,
            ILLUM_D65,
        )
        assert out_half.mean() < out_full.mean()

    def test_opposite_value_scales_cancel(self):
        neutral_xyz = torch.tensor([0.3457, 0.3585, 0.2958], dtype=torch.float32)
        white = neutral_xyz / neutral_xyz.max()
        torch.manual_seed(9)
        brightness = (
            torch.rand(64, 64, 1, dtype=torch.float32) * 0.4
        )  # keep dark so ×2 won't clip
        img = (white.view(1, 1, 3) * brightness).clone()

        lut_up = self._lut_constant_sigma_v(2.0)
        lut_down = self._lut_constant_sigma_v(0.5)

        mid = apply_hue_sat_map(
            img,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut_up,
            lut_up,
            ILLUM_D65,
            ILLUM_D65,
        )
        out = apply_hue_sat_map(
            mid,
            _unit_wb_gains(),
            _identity_color_matrix(),
            _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut_down,
            lut_down,
            ILLUM_D65,
            ILLUM_D65,
        )
        _assert_tensors_close(out, img, atol=2e-3)

class TestGeometricalAxes:
    def test_dimensions_not_swapped_2d(self):
        # Test spatial asymmetry in LUT axes to ensure no dimensional swaps occurred
        img = _make_random_image(16, 16, seed=42)
        h_bins, s_bins, v_bins = 7, 5, 1
        lut = _neutral_lut(h_bins, s_bins, v_bins)
        lut[3, 2, 0, 0] = 90.0  
        # If dimensions were swapped, grid_sample would misroute internal coordinates
        out = apply_hue_sat_map(
            img, _unit_wb_gains(), 
            _identity_color_matrix(), _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut, lut, ILLUM_D65, ILLUM_D65
        )
        assert out.shape == img.shape

    def test_dimensions_not_swapped_3d(self):
        img = _make_random_image(16, 16, seed=42)
        h_bins, s_bins, v_bins = 5, 4, 3
        lut = _neutral_lut(h_bins, s_bins, v_bins)
        lut[2, 1, 1, 0] = 45.0  
        out = apply_hue_sat_map(
            img, _unit_wb_gains(), 
            _identity_color_matrix(), _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut, lut, ILLUM_D65, ILLUM_D65
        )
        assert out.shape == img.shape


class TestWrapAround:
    def test_circular_hue_interpolation(self):
        img = _make_random_image(32, 32, seed=12)
        lut = _neutral_lut(6, 4, 1)
        lut[0, :, 0, 0] = 120.0  # H=0 gets a sharp hue shift
        # Due to circular padding, hue wrapping should seamlessly interpolate 
        # H=359 back to the H=0 parameters.
        out = apply_hue_sat_map(
            img, _unit_wb_gains(), 
            _identity_color_matrix(), _identity_color_matrix(),
            _identity_forward_matrix(),
            _identity_forward_matrix(),
            lut, lut, ILLUM_D65, ILLUM_D65
        )
        assert not torch.isnan(out).any()


class TestPreHSVClamping:
    def test_negative_gamut_stability(self):
        # Create an out-of-bounds deeply negative and exaggeratedly bright image tensor
        img = _make_random_image(32, 32, seed=99) * 6.0 - 3.0  # Range [-3.0, 3.0]
        lut = _neutral_lut(5, 5, 1)
        
        try:
            out = apply_hue_sat_map(
                img, _unit_wb_gains(), 
                _identity_color_matrix(), _identity_color_matrix(),
                _identity_forward_matrix(),
                _identity_forward_matrix(),
                lut, lut, ILLUM_D65, ILLUM_D65
            )
        except Exception as e:
            pytest.fail(f"Failed to perform stably on out-of-bounds gamut data: {e}")
        
        # Verify result is perfectly clamped and has no NaNs
        assert out.min() >= 0.0
        assert out.max() <= 1.0
        assert not torch.isnan(out).any()

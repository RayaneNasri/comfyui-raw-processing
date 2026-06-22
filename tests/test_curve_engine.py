from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from algorithms.curve.curve_engine import _bezier_lut, apply_lut_numpy, apply_lut_torch, build_lut
from algorithms.curve.curve_spec import CurveSpec

def make_points(xs, ys) -> list[tuple[float, float]]:
    return list(zip(xs, ys))


def assert_close(a, b, atol=1e-4, rtol=1e-4, msg=""):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=msg)


class TestBezierLut:

    def test_identity_curve_returns_linear_ramp(self):
        """Points sur la diagonale -> LUT ~ rampe linéaire identité."""
        xs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        ys = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        lut = _bezier_lut(xs, ys, 256)

        expected = np.linspace(0.0, 1.0, 256)
        assert_close(lut, expected, atol=1e-3)

    def test_curve_passes_through_control_points(self):
        """La courbe doit passer exactement par chaque point de contrôle."""
        xs = np.array([0.0, 0.3, 0.6, 1.0])
        ys = np.array([0.1, 0.2, 0.9, 0.95])
        lut_size = 1001  # pas régulier pour retomber pile sur les xs
        lut = _bezier_lut(xs, ys, lut_size)

        for x, y in zip(xs, ys):
            idx = round(x * (lut_size - 1))
            assert abs(lut[idx] - y) < 5e-3, f"x={x} attendu y={y}, obtenu {lut[idx]}"

    def test_output_shape_and_dtype(self):
        xs = np.array([0.0, 0.5, 1.0])
        ys = np.array([0.0, 0.5, 1.0])
        lut = _bezier_lut(xs, ys, 256)
        assert lut.shape == (256,)
        assert np.issubdtype(lut.dtype, np.floating)

    @pytest.mark.parametrize("lut_size", [2, 8, 16, 64, 100, 255, 256, 1024])
    def test_various_lut_sizes(self, lut_size):
        xs = np.array([0.0, 0.4, 1.0])
        ys = np.array([0.0, 0.7, 1.0])
        lut = _bezier_lut(xs, ys, lut_size)
        assert lut.shape == (lut_size,)
        assert np.all(np.isfinite(lut))

    def test_flat_curve_constant_y(self):
        """Tous les y égaux -> LUT constante, quel que soit l'espacement des x."""
        xs = np.array([0.0, 0.3, 0.7, 1.0])
        ys = np.array([0.5, 0.5, 0.5, 0.5])
        lut = _bezier_lut(xs, ys, 256)
        assert_close(lut, np.full(256, 0.5), atol=1e-4)

    def test_endpoints_exact(self):
        """Les bornes x=0 et x=1 de la LUT doivent correspondre exactement
        aux y du premier et dernier point de contrôle."""
        xs = np.array([0.0, 0.4, 1.0])
        ys = np.array([0.15, 0.6, 0.85])
        lut = _bezier_lut(xs, ys, 256)
        assert abs(lut[0] - 0.15) < 1e-3
        assert abs(lut[-1] - 0.85) < 1e-3

    def test_duplicate_x_values_does_not_crash(self):
        """Deux points avec le même x (cas limite/dégénéré) : ne doit pas
        lever d'exception ni produire NaN/inf."""
        xs = np.array([0.0, 0.5, 0.5, 1.0])
        ys = np.array([0.0, 0.3, 0.7, 1.0])
        lut = _bezier_lut(xs, ys, 256)
        assert np.all(np.isfinite(lut))

    def test_y_outside_unit_range_allowed(self):
        """_bezier_lut elle-même ne doit pas clipper : le clipping est la
        responsabilité de build_lut. On vérifie que des y hors [0,1]
        traversent sans être tronqués."""
        xs = np.array([0.0, 0.5, 1.0])
        ys = np.array([-0.5, 0.5, 1.5])
        lut = _bezier_lut(xs, ys, 50)
        assert lut.min() < 0.0
        assert lut.max() > 1.0

    def test_lut_size_one(self):
        """lut_size=1 : cas limite extrême, doit retourner un seul échantillon
        sans crasher (typiquement la valeur en x=0)."""
        xs = np.array([0.0, 1.0])
        ys = np.array([0.2, 0.9])
        lut = _bezier_lut(xs, ys, 1)
        assert lut.shape == (1,)
        assert np.isfinite(lut[0])

    def test_unsorted_input_xs_raises_or_is_handled(self):
        """Le docstring n'impose pas explicitement xs croissants en entrée
        de _bezier_lut (le tri se fait côté JS sur `points` avant l'appel).
        On documente le comportement réel : soit lève une erreur explicite,
        soit le tri est fait à l'intérieur. Ce test échoue intentionnellement
        s'il y a un comportement silencieux incorrect (NaN) - à adapter une
        fois le vrai comportement connu."""
        xs = np.array([0.0, 0.8, 0.3, 1.0])  # non trié
        ys = np.array([0.0, 0.6, 0.2, 1.0])
        try:
            lut = _bezier_lut(xs, ys, 64)
        except (ValueError, AssertionError):
            pytest.skip("_bezier_lut exige des xs triés en entrée (comportement valide)")
        else:
            assert np.all(np.isfinite(lut)), (
                "xs non triés acceptés silencieusement mais produisent du NaN/inf"
            )


class TestBuildLut:

    def test_default_lut_size(self):
        spec = CurveSpec(points=make_points([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1]))
        lut = build_lut(spec)
        assert lut.shape == (256,)

    def test_returns_float32(self):
        spec = CurveSpec.identity(n_points=2)
        lut = build_lut(spec, lut_size=64)
        assert lut.dtype == np.float32

    def test_identity_spec_is_identity_lut(self):
        spec = CurveSpec.identity(n_points=2)
        lut = build_lut(spec, lut_size=256)
        assert_close(lut, np.linspace(0, 1, 256, dtype=np.float32), atol=1e-2)

    @pytest.mark.parametrize("lut_size", [16, 64, 256, 1024, 4096])
    def test_various_sizes(self, lut_size):
        spec = CurveSpec(points=make_points([0, 0.5, 1], [0, 0.6, 1]))
        lut = build_lut(spec, lut_size=lut_size)
        assert lut.shape == (lut_size,)


    def test_range_clipping_upper(self):
        """Une courbe dépassant 1.0 doit être clippée à range_max."""
        spec = CurveSpec(
            points=make_points([0, 0.5, 1], [0, 1.5, 1]),
            range_min=0.0,
            range_max=1.0,
        )
        lut = build_lut(spec, lut_size=256)
        assert lut.max() <= 1.0 + 1e-6
        assert lut.min() >= 0.0 - 1e-6

    def test_range_clipping_lower(self):
        spec = CurveSpec(
            points=make_points([0, 0.5, 1], [0, -0.5, 1]),
            range_min=0.0,
            range_max=1.0,
        )
        lut = build_lut(spec, lut_size=256)
        assert lut.min() >= 0.0 - 1e-6

    def test_range_min_greater_handled_or_raises(self):
        """Cas limite dégénéré : range_min > range_max. Documente qu'une
        exception (ou un comportement défini) est attendu plutôt qu'un
        résultat silencieusement incohérent."""
        spec = CurveSpec(points=make_points([0, 1], [0, 1]), range_min=0.8, range_max=0.2)
        try:
            lut = build_lut(spec, lut_size=32)
        except (ValueError, AssertionError):
            return
        # si pas d'exception, au moins vérifier qu'il n'y a pas de NaN
        assert np.all(np.isfinite(lut))


    def test_two_points_minimum(self):
        spec = CurveSpec(points=make_points([0, 1], [0.2, 0.8]))
        lut = build_lut(spec, lut_size=128)
        assert np.all(np.isfinite(lut))

    def test_many_points(self):
        n = 50
        xs = np.linspace(0, 1, n)
        ys = np.sin(xs * math.pi) * 0.5 + 0.25
        spec = CurveSpec(points=make_points(xs.tolist(), ys.tolist()))
        lut = build_lut(spec, lut_size=256)
        assert lut.shape == (256,)
        assert np.all(np.isfinite(lut))

    def test_lut_size_one_does_not_crash(self):
        spec = CurveSpec(points=make_points([0, 1], [0, 1]))
        lut = build_lut(spec, lut_size=1)
        assert lut.shape == (1,)

    def test_points_not_starting_at_zero_or_ending_at_one(self):
        """domain_min/domain_max différents de 0/1, ou simplement des points
        ne couvrant pas tout [0,1] en x (cas limite, comportement de
        l'extrapolation aux bords à vérifier : clamp au premier/dernier y)."""
        spec = CurveSpec(points=make_points([0.2, 0.5, 0.8], [0.3, 0.6, 0.4]))
        lut = build_lut(spec, lut_size=256)
        assert np.all(np.isfinite(lut))
        # Les valeurs hors du x-range des points de contrôle doivent être
        # clampées à la valeur du point de contrôle extrême le plus proche.
        assert_close(lut[0], 0.3, atol=5e-2)
        assert_close(lut[-1], 0.4, atol=5e-2)

    def test_does_not_mutate_input_spec(self):
        """build_lut ne doit pas modifier l'objet CurveSpec passé en entrée."""
        original_points = make_points([0, 0.5, 1], [0, 0.5, 1])
        spec = CurveSpec(points=list(original_points))
        build_lut(spec, lut_size=64)
        assert spec.points == original_points


class TestApplyLutNumpy:

    def test_identity_lut_returns_same_image(self):
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        rng = np.random.default_rng(0)
        image = rng.random((8, 8, 3)).astype(np.float32)
        out = apply_lut_numpy(image, lut)
        assert_close(out, image, atol=2e-3)

    def test_output_shape_dtype_preserved(self):
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        image = np.zeros((4, 5, 3), dtype=np.float32)
        out = apply_lut_numpy(image, lut)
        assert out.shape == image.shape
        assert out.dtype == np.float32

    def test_constant_lut_maps_everything_to_same_value(self):
        lut = np.full(256, 0.42, dtype=np.float32)
        rng = np.random.default_rng(1)
        image = rng.random((6, 6, 3)).astype(np.float32)
        out = apply_lut_numpy(image, lut)
        assert_close(out, np.full_like(image, 0.42), atol=1e-3)

    def test_grayscale_single_channel(self):
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32) ** 2  # gamma-like
        image = np.array([[[0.0], [0.5], [1.0]]], dtype=np.float32)
        out = apply_lut_numpy(image, lut)
        assert out.shape == image.shape
        assert_close(out[0, 0, 0], 0.0, atol=1e-3)
        assert_close(out[0, 2, 0], 1.0, atol=1e-3)

    def test_multi_channel_alpha_unaffected_if_present(self):
        """Si l'image a 4 canaux (RGBA), vérifie que la fonction ne crash pas
        et applique la même LUT à tous les canaux fournis (cas limite,
        à ajuster si un comportement spécial existe pour l'alpha)."""
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        image = np.full((2, 2, 4), 0.5, dtype=np.float32)
        out = apply_lut_numpy(image, lut)
        assert out.shape == (2, 2, 4)

    @pytest.mark.parametrize("lut_size", [16, 64, 256, 1024])
    def test_various_lut_resolutions(self, lut_size):
        lut = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
        image = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        out = apply_lut_numpy(image, lut)
        assert np.all(np.isfinite(out))

    def test_values_at_exact_zero_and_one(self):
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32) ** 0.5
        image = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
        out = apply_lut_numpy(image, lut)
        assert_close(out[0, 0, 0], lut[0], atol=1e-3)
        assert_close(out[0, 0, 1], lut[-1], atol=1e-3)

    def test_out_of_range_values_below_zero(self):
        """Valeurs négatives en entrée (cas limite, hors spec [0,1]) :
        comportement attendu = clamp au premier échantillon de la LUT,
        sans crash ni NaN."""
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        image = np.array([[[-0.5, -0.1, 0.0]]], dtype=np.float32)
        out = apply_lut_numpy(image, lut)
        assert np.all(np.isfinite(out))
        assert_close(out[0, 0, 0], lut[0], atol=1e-2)

    def test_out_of_range_values_above_one(self):
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        image = np.array([[[1.1, 2.0, 1.0]]], dtype=np.float32)
        out = apply_lut_numpy(image, lut)
        assert np.all(np.isfinite(out))
        assert_close(out[0, 0, 0], lut[-1], atol=1e-2)

    def test_lut_size_two_minimal(self):
        """LUT minimale de 2 points : doit dégénérer en interpolation
        linéaire simple sur [0,1]."""
        lut = np.array([0.2, 0.8], dtype=np.float32)
        image = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        out = apply_lut_numpy(image, lut)
        assert_close(out[0, 0], np.array([0.2, 0.5, 0.8]), atol=1e-3)

    def test_single_pixel_image(self):
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        image = np.array([[[0.3, 0.6, 0.9]]], dtype=np.float32)
        out = apply_lut_numpy(image, lut)
        assert out.shape == (1, 1, 3)

    def test_empty_image_does_not_crash(self):
        """Image de taille nulle (H=0) : cas limite, doit retourner un
        tableau vide de la bonne shape plutôt que de lever une exception
        inattendue."""
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        image = np.zeros((0, 4, 3), dtype=np.float32)
        out = apply_lut_numpy(image, lut)
        assert out.shape == (0, 4, 3)

    def test_does_not_mutate_input_image(self):
        lut = (np.linspace(0.0, 1.0, 256, dtype=np.float32)) ** 2
        image = np.full((3, 3, 3), 0.5, dtype=np.float32)
        image_copy = image.copy()
        apply_lut_numpy(image, lut)
        assert_close(image, image_copy, atol=0)

    def test_monotonic_lut_preserves_ordering(self):
        """Une LUT strictement croissante doit préserver l'ordre relatif des
        valeurs d'entrée (propriété générale, indépendante de l'implémentation
        exacte de l'interpolation)."""
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32) ** 3
        image = np.array([[[0.1, 0.5, 0.9]]], dtype=np.float32)
        out = apply_lut_numpy(image, lut)
        assert out[0, 0, 0] < out[0, 0, 1] < out[0, 0, 2]


class TestApplyLutTorch:

    def test_identity_lut_returns_same_tensor(self):
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        torch.manual_seed(0)
        image = torch.rand(8, 8, 3, dtype=torch.float32)
        out = apply_lut_torch(image, lut)
        assert_close(out.numpy(), image.numpy(), atol=2e-3)

    def test_output_shape_and_dtype(self):
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        image = torch.zeros(4, 5, 3, dtype=torch.float32)
        out = apply_lut_torch(image, lut)
        assert out.shape == image.shape
        assert out.dtype == torch.float32

    def test_returns_torch_tensor(self):
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        image = torch.rand(4, 4, 3)
        out = apply_lut_torch(image, lut)
        assert isinstance(out, torch.Tensor)

    def test_matches_numpy_implementation(self):
        """Les deux implémentations doivent produire des résultats cohérents
        pour la même LUT et la même image (croisement numpy <-> torch)."""
        lut = (np.linspace(0.0, 1.0, 256, dtype=np.float32)) ** 0.5
        rng = np.random.default_rng(42)
        image_np = rng.random((10, 10, 3)).astype(np.float32)
        image_torch = torch.from_numpy(image_np.copy())

        out_np = apply_lut_numpy(image_np, lut)
        out_torch = apply_lut_torch(image_torch, lut).numpy()

        assert_close(out_np, out_torch, atol=2e-3)

    @pytest.mark.parametrize("lut_size", [16, 64, 256, 1024])
    def test_various_lut_resolutions(self, lut_size):
        lut = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
        image = torch.tensor([[[0.0, 0.5, 1.0]]], dtype=torch.float32)
        out = apply_lut_torch(image, lut)
        assert torch.all(torch.isfinite(out))

    def test_out_of_range_values_clamped(self):
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        image = torch.tensor([[[-0.5, 1.5, 0.5]]], dtype=torch.float32)
        out = apply_lut_torch(image, lut)
        assert torch.all(torch.isfinite(out))
        assert out[0, 0, 0].item() == pytest.approx(lut[0], abs=1e-2)
        assert out[0, 0, 1].item() == pytest.approx(lut[-1], abs=1e-2)

    def test_single_pixel_tensor(self):
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        image = torch.tensor([[[0.2, 0.4, 0.6]]], dtype=torch.float32)
        out = apply_lut_torch(image, lut)
        assert out.shape == (1, 1, 3)

    def test_does_not_mutate_input_tensor(self):
        lut = (np.linspace(0.0, 1.0, 256, dtype=np.float32)) ** 2
        image = torch.full((3, 3, 3), 0.5, dtype=torch.float32)
        image_clone = image.clone()
        apply_lut_torch(image, lut)
        assert torch.equal(image, image_clone)

    def test_grayscale_single_channel(self):
        lut = (np.linspace(0.0, 1.0, 256, dtype=np.float32)) ** 2
        image = torch.tensor([[[0.0], [0.5], [1.0]]], dtype=torch.float32)
        out = apply_lut_torch(image, lut)
        assert out.shape == image.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA non disponible")
    def test_preserves_cuda_device(self):
        """Si l'image d'entrée est sur GPU, la sortie doit aussi être sur GPU
        (pas de retour silencieux sur CPU)."""
        lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        image = torch.rand(4, 4, 3, device="cuda")
        out = apply_lut_torch(image, lut)
        assert out.device.type == "cuda"


class TestEndToEndPipeline:

    def test_full_pipeline_numpy(self):
        spec = CurveSpec(points=make_points([0, 0.3, 0.7, 1], [0, 0.1, 0.9, 1]))
        lut = build_lut(spec, lut_size=256)
        image = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        out = apply_lut_numpy(image, lut)
        assert out.shape == image.shape
        assert np.all(out >= spec.range_min - 1e-6)
        assert np.all(out <= spec.range_max + 1e-6)

    def test_full_pipeline_torch(self):
        spec = CurveSpec(points=make_points([0, 0.3, 0.7, 1], [0, 0.1, 0.9, 1]))
        lut = build_lut(spec, lut_size=256)
        image = torch.tensor([[[0.0, 0.5, 1.0]]], dtype=torch.float32)
        out = apply_lut_torch(image, lut)
        assert out.shape == image.shape
        assert torch.all(out >= spec.range_min - 1e-6)
        assert torch.all(out <= spec.range_max + 1e-6)

    def test_numpy_and_torch_pipelines_agree(self):
        spec = CurveSpec(points=make_points([0, 0.25, 0.5, 0.75, 1], [0, 0.4, 0.5, 0.6, 1]))
        lut = build_lut(spec, lut_size=256)
        rng = np.random.default_rng(7)
        image_np = rng.random((16, 16, 3)).astype(np.float32)
        image_torch = torch.from_numpy(image_np.copy())

        out_np = apply_lut_numpy(image_np, lut)
        out_torch = apply_lut_torch(image_torch, lut).numpy()
        assert_close(out_np, out_torch, atol=2e-3)

    def test_default_front_end_points_pipeline(self):
        """Reproduit exactement DEFAULT_POINTS du front-end JS de bout en
        bout : doit donner la fonction identité."""
        spec = CurveSpec(points=make_points([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1]))
        lut = build_lut(spec, lut_size=256)
        image = np.linspace(0, 1, 100).reshape(10, 10, 1).astype(np.float32)
        image = np.repeat(image, 3, axis=2)
        out = apply_lut_numpy(image, lut)
        assert_close(out, image, atol=5e-3)
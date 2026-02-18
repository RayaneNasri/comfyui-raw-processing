import torch

from src.algorithms.malvar_he_culter import malvar_he_cutler_demosaicing


def _cfa_masks(height: int, width: int, dx: int, dy: int):
    yy = torch.arange(height).view(-1, 1)
    xx = torch.arange(width).view(1, -1)

    red_mask = ((yy % 2) == dy) & ((xx % 2) == dx)
    blue_mask = ((yy % 2) == (1 - dy)) & ((xx % 2) == (1 - dx))
    green_mask = ~(red_mask | blue_mask)

    return red_mask, green_mask, blue_mask


def test_empty_array():
    out = malvar_he_cutler_demosaicing(torch.empty((0, 0), dtype=torch.float32), 0, 0)
    assert out.numel() == 0
    assert out.shape == (0, 0, 3)


def test_single_pixel_no_neighbors():
    raw = torch.zeros((1, 1), dtype=torch.float32)
    raw[0, 0] = 100
    out = malvar_he_cutler_demosaicing(raw, 0, 0)

    assert out[0, 0, 0] == 100
    assert out[0, 0, 1] == 0
    assert out[0, 0, 2] == 0


def test_green_interpolation_at_red_location():
    raw = torch.zeros((3, 3), dtype=torch.float32)
    raw[1, 1] = 50
    raw[0, 1] = 100
    raw[2, 1] = 200
    raw[1, 0] = 300
    raw[1, 2] = 400

    out = malvar_he_cutler_demosaicing(raw, 1, 1)

    assert out[1, 1, 1] == 250
    assert out[1, 1, 0] == 50


def test_blue_interpolation_at_red_location():
    raw = torch.zeros((3, 3), dtype=torch.float32)
    raw[1, 1] = 50
    raw[0, 0] = 10
    raw[0, 2] = 20
    raw[2, 0] = 30
    raw[2, 2] = 40

    out = malvar_he_cutler_demosaicing(raw, 1, 1)

    assert out[1, 1, 2] == 25


def test_red_interpolation_at_green_location_red_row():
    raw = torch.zeros((5, 5), dtype=torch.float32)
    raw[2, 1] = 100
    raw[2, 3] = 200

    out = malvar_he_cutler_demosaicing(raw, 1, 0)

    assert out[2, 2, 0] == 150


def test_red_interpolation_at_green_location_blue_row():
    raw = torch.zeros((5, 5), dtype=torch.float32)
    raw[1, 2] = 100
    raw[3, 2] = 300

    out = malvar_he_cutler_demosaicing(raw, 0, 1)

    assert out[2, 2, 0] == 200


def test_corner_edge_cases():
    raw = torch.zeros((2, 2), dtype=torch.float32)
    raw[0, 1] = 100
    raw[1, 0] = 200

    out = malvar_he_cutler_demosaicing(raw, 0, 0)

    assert out[0, 0, 1] == 150


def test_edge_cases_constant_field():
    raw = torch.ones((4, 3), dtype=torch.float32)
    out = malvar_he_cutler_demosaicing(raw, 0, 0)

    assert torch.isfinite(out).all()
    red_mask, green_mask, blue_mask = _cfa_masks(4, 3, 0, 0)
    assert torch.allclose(out[..., 0][red_mask], raw[red_mask])
    assert torch.allclose(out[..., 1][green_mask], raw[green_mask])
    assert torch.allclose(out[..., 2][blue_mask], raw[blue_mask])


def test_data_types():
    raw = torch.zeros((5, 5), dtype=torch.float64)
    raw[2, 1] = 0.5
    raw[2, 3] = 1.0

    out = malvar_he_cutler_demosaicing(raw, 1, 0)

    assert out[2, 2, 0] == 0.75
    assert out.dtype == torch.float64


def test_sampled_channel_preservation_for_all_offsets():
    height, width = 6, 6
    raw = torch.arange(height * width, dtype=torch.float32).reshape(height, width)

    for dy in (0, 1):
        for dx in (0, 1):
            out = malvar_he_cutler_demosaicing(raw, dx=dx, dy=dy)
            red_mask, green_mask, blue_mask = _cfa_masks(height, width, dx, dy)

            assert torch.allclose(out[..., 0][red_mask], raw[red_mask])
            assert torch.allclose(out[..., 1][green_mask], raw[green_mask])
            assert torch.allclose(out[..., 2][blue_mask], raw[blue_mask])


def test_full_image_constant_sanity():
    raw = torch.ones((8, 8), dtype=torch.float32)
    out = malvar_he_cutler_demosaicing(raw, 0, 0)

    interior = out[2:-2, 2:-2, :]
    assert torch.allclose(interior, torch.ones_like(interior))

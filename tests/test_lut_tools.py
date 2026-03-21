import torch

from algorithms.tools._lut_tools import rgb_to_hsv, hsv_to_rgb


def test_rgb_hsv_identity():
    img = torch.rand(4, 5, 3)
    out = hsv_to_rgb(rgb_to_hsv(img))
    assert torch.allclose(out, img, atol=1e-6)


def test_rgb_hsv_single_channel():
    red = torch.tensor([[[1.0, 0.0, 0.0]]])
    green = torch.tensor([[[0.0, 1.0, 0.0]]])
    blue = torch.tensor([[[0.0, 0.0, 1.0]]])
    black = torch.tensor([[[0.0, 0.0, 0.0]]])
    white = torch.tensor([[[1.0, 1.0, 1.0]]])

    red_hsv = rgb_to_hsv(red)[0, 0]
    green_hsv = rgb_to_hsv(green)[0, 0]
    blue_hsv = rgb_to_hsv(blue)[0, 0]
    black_hsv = rgb_to_hsv(black)[0, 0]
    white_hsv = rgb_to_hsv(white)[0, 0]

    print(green, green_hsv)

    assert torch.allclose(red_hsv, torch.tensor([0.0, 1.0, 1.0]), atol=1e-6)
    assert torch.allclose(green_hsv, torch.tensor([1 / 3, 1.0, 1.0]), atol=1e-6)
    assert torch.allclose(blue_hsv, torch.tensor([2 / 3, 1.0, 1.0]), atol=1e-6)
    assert torch.allclose(black_hsv, torch.tensor([0.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(white_hsv, torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)

    assert torch.allclose(hsv_to_rgb(red_hsv.unsqueeze(0).unsqueeze(0)), red, atol=1e-6)
    assert torch.allclose(
        hsv_to_rgb(green_hsv.unsqueeze(0).unsqueeze(0)), green, atol=1e-6
    )
    assert torch.allclose(
        hsv_to_rgb(blue_hsv.unsqueeze(0).unsqueeze(0)), blue, atol=1e-6
    )
    assert torch.allclose(
        hsv_to_rgb(black_hsv.unsqueeze(0).unsqueeze(0)), black, atol=1e-6
    )
    assert torch.allclose(
        hsv_to_rgb(white_hsv.unsqueeze(0).unsqueeze(0)), white, atol=1e-6
    )


def test_hsv_to_rgb_zero_saturation_is_grey():
    hues = torch.linspace(0, 360, 37)
    vals = torch.linspace(0.0, 1.0, 11)
    h, v = torch.meshgrid(hues, vals, indexing="ij")
    s = torch.zeros_like(h)
    image_hsv = torch.stack([h, s, v], dim=-1)
    rgb = hsv_to_rgb(image_hsv)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    assert torch.allclose(r, v, atol=1e-6)
    assert torch.allclose(g, v, atol=1e-6)
    assert torch.allclose(b, v, atol=1e-6)

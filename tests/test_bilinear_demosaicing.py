import torch

from algorithms.demosaicing import bilinear_demosaicing

# Bayer CFA with (dy,dx) the location of the first pixel sampled in red
# pixels in rgb_image[dy::2, dx::2] are sampled in red
# those in rgb_image[1-dy::2, 1-dx::2] in blue
# those in rgb_image[dy::2, 1-dx::2] and in rgb_image[1-dy::2, dx::2] in green.
# Example of Bayer CFA with (dy,dx) = (0,0):
# R G R G R
# G B G B G
# R G R G R


def test_empty_tensor():
    assert bilinear_demosaicing(torch.empty(0, 0, 3)).nelement() == 0


def test_single_pixel_no_neighbors():
    img = torch.zeros((1, 1, 3))
    img[0, 0, 0] = 100  # Red present
    res = bilinear_demosaicing(img)
    # The existing value should be preserved
    assert res[0, 0, 0] == 100
    # Missing channels with no neighbors remain 0
    assert res[0, 0, 1] == 0
    assert res[0, 0, 2] == 0


def test_green_interpolation_at_red_location():
    """
    Test Green channel interpolation at a Red pixel location (Red at center).
    Input Grid (3x3):
      . G .
      G R G
      . G .
    """
    img = torch.zeros((3, 3, 3))
    # Set center Red
    img[1, 1, 0] = 50

    # Set surrounding Green neighbors (up, down, left, right)
    img[0, 1, 1] = 100  # Top
    img[2, 1, 1] = 200  # Bottom
    img[1, 0, 1] = 300  # Left
    img[1, 2, 1] = 400  # Right

    res = bilinear_demosaicing(img, 1, 1)

    # Expected: Average of 100, 200, 300, 400 = 1000 / 4 = 250
    assert res[1, 1, 1] == 250
    # Center Red should stay
    assert res[1, 1, 0] == 50


def test_blue_interpolation_at_red_location():
    """
    Test Blue channel interpolation at a Red pixel location.
    Standard bilinear uses diagonal neighbors.
    Input Grid (3x3):
      B . B
      . R .
      B . B
    """
    img = torch.zeros((3, 3, 3))
    # Set center Red
    img[1, 1, 0] = 50

    # Set surrounding Blue neighbors (diagonals)
    img[0, 0, 2] = 10
    img[0, 2, 2] = 20
    img[2, 0, 2] = 30
    img[2, 2, 2] = 40

    res = bilinear_demosaicing(img, 1, 1)

    # Expected: Average of 10, 20, 30, 40 = 100 / 4 = 25
    assert res[1, 1, 2] == 25


def test_red_interpolation_at_green_location_red_row():
    """
    Test Red interpolation at a Green pixel (on a Red row).
    Input Grid (3x3):
      . B .
      R G R
      . B .
    """
    img = torch.zeros((3, 3, 3))
    # Center Green
    img[1, 1, 1] = 50

    # Red neighbors (Left, Right)
    img[1, 0, 0] = 100
    img[1, 2, 0] = 200

    res = bilinear_demosaicing(img, 1, 0)

    # Expected Red: Average of 100, 200 = 150
    assert res[1, 1, 0] == 150


def test_red_interpolation_at_green_location_blue_row():
    """
    Test Red interpolation at a Green pixel (on a Blue row).
    Input Grid (3x3):
      . R .
      B G B  <-- Center is Green, rows above/below have Red
      . R .
    """
    img = torch.zeros((3, 3, 3))
    # Center Green
    img[1, 1, 1] = 50

    # Red neighbors (Top, Bottom)
    img[0, 1, 0] = 100
    img[2, 1, 0] = 300

    res = bilinear_demosaicing(img, 0, 1)

    # Expected Red: Average of 100, 300 = 200
    assert res[1, 1, 0] == 200


def test_corner_edge_cases():
    """
    Test interpolation at a corner (0,0).
    Grid 2x2:
      R G
      G B
    G Neighbors of (0,0) are (0,1) and (1,0).
    """
    img = torch.zeros((2, 2, 3))

    # Set neighbors for Green
    img[0, 1, 1] = 100  # Right
    img[1, 0, 1] = 200  # Bottom

    res = bilinear_demosaicing(img, 0, 0)

    # At (0,0), valid G neighbors are just Right and Bottom.
    # Average = (100 + 200) / 2 = 150
    assert res[0, 0, 1] == 150


def test_edge_cases():
    """Test edge interpolation avec une grille 4x3."""
    img = torch.zeros((4, 3, 3))
    img[::2, ::2, 0] = 1
    img[1::2, 1, 2] = 1
    img[0::2, 1, 1] = 1
    img[1::2, ::2, 1] = 1
    res = bilinear_demosaicing(img, 0, 0)
    expected = torch.ones_like(res)
    torch.testing.assert_close(res, expected)


def test_data_types():
    """
    Ensure floats are handled correctly.
    . . .
    R . R
    . . .
    """
    img = torch.zeros((3, 3, 3), dtype=torch.float32)
    img[1, 0, 0] = 0.5
    img[1, 2, 0] = 1.0
    res = bilinear_demosaicing(img, 1, 0)
    # Center (1,1,0) should be 0.75
    assert res[1, 1, 0] == 0.75
    assert res.dtype == torch.float32

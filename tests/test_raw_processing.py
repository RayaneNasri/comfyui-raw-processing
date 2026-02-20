import torch
from algorithms.raw_processing import mono_to_rgb


def test_mono_to_rgb_basic_rggb():
    """
    Tests the demosaicing logic on a simple 2x2 RGGB pattern.
    Verifies that pixels are mapped to their specific color channels
    and other channels remain zero.
    """
    # 1. Setup Data
    # A simple 2x2 normalized image
    # [ 0.1, 0.2 ]
    # [ 0.3, 0.4 ]
    normalized_image = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)

    # A Bayer pattern (e.g., RGGB)
    # 0=Red, 1=Green, 2=Blue
    # [ R, G ] -> [ 0, 1 ]
    # [ G, B ] -> [ 1, 2 ]
    bayer_pattern = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    # 2. Execute
    output = mono_to_rgb(normalized_image, bayer_pattern)

    # 3. Validation

    # Check Shape: Should be (3, Height, Width)
    assert output.shape == (2, 2, 3), f"Expected shape (3, 2, 2), got {output.shape}"

    # Expected Red Channel (Channel 0)
    # Only index (0,0) is Red (0) in bayer_pattern
    expected_r = torch.tensor([[0.1, 0.0], [0.0, 0.0]])
    torch.testing.assert_close(output[:, :, 0], expected_r, msg="Red channel mismatch")

    # Expected Green Channel (Channel 1)
    # Indices (0,1) and (1,0) are Green (1)
    expected_g = torch.tensor([[0.0, 0.2], [0.3, 0.0]])
    torch.testing.assert_close(
        output[:, :, 1], expected_g, msg="Green channel mismatch"
    )

    # Expected Blue Channel (Channel 2)
    # Only index (1,1) is Blue (2)
    expected_b = torch.tensor([[0.0, 0.0], [0.0, 0.4]])
    torch.testing.assert_close(output[:, :, 2], expected_b, msg="Blue channel mismatch")

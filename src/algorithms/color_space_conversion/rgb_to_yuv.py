import torch

def rgb_to_yuv(rgb_image: torch.Tensor) -> torch.Tensor:
    """Converts an RGB image tensor to YUV color space."""
    assert rgb_image.ndim == 3, "not a 3D image tensor"
    # Ensuring the channel dimension is at the end (H, W, 3) to match np.dot behavior
    assert rgb_image.shape[-1] == 3, "expected channel dimension to be 3 (H, W, 3)"
    assert rgb_image.is_floating_point(), "expected a float image tensor"
    assert rgb_image.min() >= 0.0 and rgb_image.max() <= 1.0, (
        "expected float image between 0.0 and 1.0"
    )

    # Create the matrix on the same device and dtype as the image tensor
    cv_mat = torch.tensor(
        [
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001],
        ],
        device=rgb_image.device,
        dtype=rgb_image.dtype,
    )

    # In PyTorch, @ performs matrix multiplication (equivalent to np.dot on the last dim)
    return torch.clamp(rgb_image @ cv_mat.T, 0.0, 1.0)
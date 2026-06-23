import cv2
import numpy as np
import torch


def correct_distortion(
    image: torch.Tensor,
    k1: float,
    k2: float,
    p1: float = 0.0,
    p2: float = 0.0,
) -> torch.Tensor:
    """
    Correct radial and tangential lens distortion using the Brown-Conrady model.

    The camera intrinsics are assumed: principal point at the image centre and
    focal length equal to max(H, W).  These are reasonable defaults when the
    true intrinsics are unknown.  The correction is applied via `cv2.undistort`.

    Args:
        image: (H, W, 3) linear RGB tensor in [0, 1].
        k1:    First radial distortion coefficient.
               Negative → barrel distortion (wide-angle), positive → pincushion.
        k2:    Second radial distortion coefficient (higher-order term).
        p1:    First tangential distortion coefficient.  Usually near zero.
        p2:    Second tangential distortion coefficient.  Usually near zero.

    Returns:
        (H, W, 3) corrected image clamped to [0, 1].
    """
    if k1 == 0.0 and k2 == 0.0 and p1 == 0.0 and p2 == 0.0:
        return image

    H, W, _ = image.shape
    f = float(max(H, W))
    camera_matrix = np.array(
        [[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float32)

    img_np = image.numpy().astype(np.float32)
    corrected = cv2.undistort(img_np, camera_matrix, dist_coeffs)
    return torch.from_numpy(corrected).clamp(0.0, 1.0)

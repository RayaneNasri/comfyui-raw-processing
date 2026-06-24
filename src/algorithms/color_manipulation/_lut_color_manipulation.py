import torch
from torch import Tensor


def load_cube_lut(path):
    size = None
    data = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                continue

            if line.startswith("DOMAIN_"):
                continue

            # Data line
            values = list(map(float, line.split()))
            if len(values) == 3:
                data.append(values)

    data = torch.tensor(data, dtype=torch.float32)

    if size is None:
        raise ValueError("LUT_3D_SIZE not found")

    lut = data.view(size, size, size, 3)

    return lut


def linearRGB_to_adobeRGB1998(image: Tensor) -> Tensor:
    """
    image: Tensor Linear_RGB image (H,W,3) with each channel represented as a float in [0,1]

    return: Tensor AdobeRGB1998 image (H,W,3) with each channel represented as a float in [0,1]
    """
    gamma = 1 / 2.19921875
    return torch.pow(image, gamma)


def adobeRGB1998_to_linearRGB(image: Tensor) -> Tensor:
    """
    image: Tensor AdobeRGB1998 image (H,W,3) with each channel represented as a float in [0,1]

    return: Tensor Linear_RGB image (H,W,3) with each channel represented as a float in [0,1]
    """
    gamma = 1 / 2.19921875
    return torch.pow(image, (1 / gamma))


def apply_lut_grid_sample(image: Tensor, lut: Tensor) -> Tensor:
    """
    - image: Tensor rgb image (H,W,3) with each channel represented as a float in [0,1]
    - lut: Tensor rgb (S,S,S,3)
    image and lut must be in the same color-space
    """

    # image shape
    if image.dim() == 3:
        _, _, C = image.shape
        image = image.unsqueeze(0)  # add batch
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if C != 3:
        raise ValueError(f"Image must have 3 channels, got {C}")

    # grid in [-1,1]
    grid = torch.clamp(image * 2 - 1, -1, 1)
    # RGB -> BGR for grid_sample coordinate order (x,y,z)
    grid = grid[..., [2, 1, 0]]
    # add depth dimension
    grid = grid.unsqueeze(1)  # (B,1,H,W,3)

    # LUT -> (N,C,D,H,W)
    lut = lut.permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,S,S,S)

    out = torch.nn.functional.grid_sample(
        lut, grid, mode="bilinear", align_corners=True
    )

    # remove depth dimension
    out = out.squeeze(2)  # (B,3,H,W)
    # back to (B,H,W,3)
    out = out.permute(0, 2, 3, 1)

    # back to (H,W,3)
    out = out.squeeze(0)

    return out

import torch
from torch import Tensor

def load_cube_lut(path):
    size = None
    data = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            if line.startswith('LUT_3D_SIZE'):
                size = int(line.split()[-1])
                continue

            if line.startswith('DOMAIN_'):
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

def apply_lut_grid_sample(image : Tensor , lut : Tensor) -> Tensor:
    """
    image: (H,W,3) or (B,H,W,3) or (B,3,H,W) float tensor in [0,1]
    lut:   (S,S,S,3)
    """
    
    # ---- Normalisation du format ----
    if image.dim() == 4:
        # Cas (B,3,H,W) → (B,H,W,3)
        if image.shape[1] == 3:
            image = image.permute(0, 2, 3, 1)

        B, H, W, C = image.shape

    elif image.dim() == 3:
        H, W, C = image.shape
        image = image.unsqueeze(0)  # add batch
        B = 1

    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if C != 3:
        raise ValueError(f"Image must have 3 channels, got {C}")
        
    S = lut.shape[0]

    # from linearRGB to AdobeRGB1998
    gamma = 1/2.19921875
    image_adobe_rgb = torch.pow(image, gamma)

    # grid in [-1,1]
    grid = torch.clamp(image_adobe_rgb * 2 - 1, -1, 1)
    # RGB -> BGR for grid_sample coordinate order (x,y,z)
    grid = grid[..., [2,1,0]]
    # add depth dimension
    grid = grid.unsqueeze(1) # (B,1,H,W,3)

    # LUT -> (N,C,D,H,W)
    lut = lut.permute(3,0,1,2).unsqueeze(0)  # (1,3,S,S,S)

    out = torch.nn.functional.grid_sample(lut, grid, mode='bilinear', align_corners=True)
    
    # remove depth dimension
    out = out.squeeze(2)           # (B,3,H,W)
    # back to (B,H,W,3)
    out = out.permute(0,2,3,1)
    # BGR-> RGB
    out = out[..., [2,1,0]]

    # from AdobeRGB1998 to linearRGB
    out = torch.pow(out, (1/gamma))

    out = out.squeeze(0) # (H,W,3) TODO : keep it ?

    return out
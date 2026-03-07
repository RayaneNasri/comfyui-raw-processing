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

    # grid in [-1,1]
    grid = image * 2 - 1
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

    return out

def apply_lut_trilinear_interpolation(image, lut):
    """
    image: (H,W,3) or (B,H,W,3) or (B,3,H,W) float tensor in [0,1]
    lut:   (S,S,S,3)
    """

    try:
        # ---- Normalisation du format ----
        if image.dim() == 4:
            # Cas (B,3,H,W) → (B,H,W,3)
            if image.shape[1] == 3:
                image = image.permute(0, 2, 3, 1)

            B, H, W, C = image.shape

        elif image.dim() == 3:
            H, W, C = image.shape
            image = image.unsqueeze(0)  # ajout batch
            B = 1

        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if C != 3:
            raise ValueError(f"Image must have 3 channels, got {C}")

        # ---- Trilinear Interpolation ----
        S = lut.shape[0]

        # Scale to LUT coordinates
        # Your image values are in [0, 1], but LUT grid indices are [0, S-1]
        coords = image * (S - 1)

        # Find surrounding grid points (floor indices idx0 and ceil indices idx1) and interpolation weights (frac)
        # for each pixel, if coords = (15.3, 10.8, 2.2),
        # then idx0 = (15, 10, 2), idx1 = (16, 11, 3) et frac = (0.3, 0.8, 0.2)
        idx0 = torch.floor(coords).long()
        idx1 = torch.clamp(idx0 + 1, max=S - 1)

        frac = coords - idx0.float()

        # for each pixel, the two points (r0, g0, b0) and (r1, g1, b1) define a 2*2*2 cube 
        r0, g0, b0 = idx0[..., 0], idx0[..., 1], idx0[..., 2]
        r1, g1, b1 = idx1[..., 0], idx1[..., 1], idx1[..., 2]

        # fr : how far along red axis
        # fg : how far along green axis
        # fb : how far along blue axis
        fr, fg, fb = frac[..., 0], frac[..., 1], frac[..., 2]

        # Fetch 8 neighbors (corners of the 2*2*2 cube)
        c000 = lut[r0, g0, b0]
        c001 = lut[r0, g0, b1]
        c010 = lut[r0, g1, b0]
        c011 = lut[r0, g1, b1]
        c100 = lut[r1, g0, b0]
        c101 = lut[r1, g0, b1]
        c110 = lut[r1, g1, b0]
        c111 = lut[r1, g1, b1]

        # Interpolate between the 8 surrounding cube points
        c00 = c000 * (1 - fb)[..., None] + c001 * fb[..., None]
        c01 = c010 * (1 - fb)[..., None] + c011 * fb[..., None] 
        c10 = c100 * (1 - fb)[..., None] + c101 * fb[..., None]
        c11 = c110 * (1 - fb)[..., None] + c111 * fb[..., None] 

        c0 = c00 * (1 - fg)[..., None] + c01 * fg[..., None]
        c1 = c10 * (1 - fg)[..., None] + c11 * fg[..., None]

        output = c0 * (1 - fr)[..., None] + c1 * fr[..., None]

        return output

    except Exception as e:
        print("Error in apply_lut")
        print("Image shape:", image.shape)
        print("LUT shape:", lut.shape)
        raise e
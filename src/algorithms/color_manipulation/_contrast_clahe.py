import torch
from torch import Tensor 

import numpy as np
import cv2
import kornia as K

def contrast_clahe_cv2(rgb_image : Tensor, clip_limit : float = 2.0, grid_size : tuple[int, int] = (8, 8)) -> Tensor:

    """
    Change the default grid_size to (16, 16) ?
    """

    device = rgb_image.device
    dtype = rgb_image.dtype

    # Tensor -> uint8 numpy
    img = (rgb_image.cpu().numpy() * 255).astype(np.uint8)

    # RGB -> LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # canal separation
    L, A, B = cv2.split(lab)

    # CLAHE on L
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=grid_size
    )

    L_eq = clahe.apply(L)

    # reconstruction LAB
    lab_eq = cv2.merge((L_eq, A, B))

    # LAB -> RGB
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # return tensor float in [0,1]
    out = (
        torch.from_numpy(rgb_eq)
        .to(device=device, dtype=dtype)
        / 255.0
    ).clamp(0.0,1.0)

    return out




def contrast_clahe_kornia(rgb_image : Tensor, clip_limit : float = 4.0, grid_size : tuple[int, int] = (8, 8)) -> Tensor:
    """

    kornia.enhance.equalize_clahe(rgb_image, clip_limit, grid_size) applique une CLAHE (Contrast Limited Adaptive Histogram Equalization)
    image par image, canal par canal, sur un tenseur PyTorch normalisé dans [0,1][0,1] et de forme (C,H,W)(C,H,W) ou (B,C,H,W)(B,C,H,W).
    Kornia précise que la fonction divise l’image en tuiles selon grid_size, calcule un histogramme local par tuile, limite le contraste via clip_limit,
    construit une LUT (look-up table) par tuile, puis reconstruit l’image avec interpolation bilinéaire entre tuiles, comme dans l’approche OpenCV

    CLAHE est défini pour une image en niveaux de gris. Évitez d’appliquer CLAHE séparément sur R, G et B, car cela peut déformer les couleurs. Mieux vaut travailler sur la luminance.

    clip_limit plus petit : effet plus doux. threshold value for contrast limiting. If 0 clipping is disabled. Default: 40.0

    grid_size plus grand : contraste local plus global. number of tiles to be cropped in each direction (GH, GW). Default: (8, 8)

    Valeurs typiques de départ : clipLimit=2.0, tileGridSize=(8, 8) pour OpenCV, ou les valeurs par défaut de Kornia selon votre pipeline.
    """
    
    # (H,W,3) -> (1,3,H,W)
    rgb_image = rgb_image.permute(2, 0, 1).unsqueeze(0)

    # RGB -> Lab
    lab = K.color.rgb_to_lab(rgb_image)

    # canal luminance and normalisation
    L = lab[:, :1, :, :] # in [0,100]
    L_norm = (L/100.0).clamp(0.0,1.0) # in [0,1]

    # CLAHE
    L_eq = K.enhance.equalize_clahe(L_norm, clip_limit, grid_size) #Ici, equalize_clahe ne voit qu’un seul canal, donc elle ne modifie que la luminance.

    # Reconstruction Lab
    L_eq = (L_eq * 100.).clamp(0., 100.) # in [0,100]
    lab_eq = torch.cat([L_eq, lab[:, 1:, :, :]], dim=1)

    # Lab -> RGB
    rgb_eq = K.color.lab_to_rgb(lab_eq)

    # (1,3,H,W) -> (H,W,3)
    rgb_eq = rgb_eq.squeeze(0).permute(1, 2, 0)
    
    return rgb_eq.clamp(0.0, 1.0)

    #rgb_image = rgb_image.permute(2, 0, 1).unsqueeze(0)  # -> (1, 3, H, W)
    #y = kornia.enhance.equalize_clahe(rgb_image, clip_limit, grid_size)
    #return y.squeeze(0).permute(1, 2, 0)



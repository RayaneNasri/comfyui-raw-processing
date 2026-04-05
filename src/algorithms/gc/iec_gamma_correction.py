import torch

def iec_gamma_correction(img: torch.Tensor) -> torch.Tensor:
    """
    Applique la correction gamma standard sRGB (moderne et universelle).
    Compatible avec les attentes du pipeline de Karaimer et les standards Web/Rec.709.
    
    Parameters
    ----------
    img : torch.Tensor (Linear RGB, range [0, 1])
    """
    img = torch.clamp(img, 0.0, 1.0)
    out = torch.empty_like(img)
    mask = img <= 0.0031308
    out[mask] = 12.92 * img[mask]
    out[~mask] = 1.055 * torch.pow(img[~mask], 1/2.4) - 0.055
    
    return torch.clamp(out, 0.0, 1.0)
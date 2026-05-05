import torch


def iec_gamma_correction(img: torch.Tensor) -> torch.Tensor:
    """
    Applique la correction gamma standard sRGB (moderne et universelle).
    Compatible avec les attentes du pipeline de Karaimer et les standards Web/Rec.709.

    Parameters
    ----------
    img : torch.Tensor (Linear RGB, range [0, 1])
    """
    img = img.clamp_(0.0, 1.0)         # in-place — no extra allocation
 
    out  = torch.empty_like(img)       # uninitialized — every element will be written
    mask = img <= 0.0031308            # single bool tensor [H, W, 3]
 
    out[ mask] = img[ mask] * 12.92
    out[~mask] = 1.055 * torch.pow(img[~mask], 1.0 / 2.4) - 0.055
 
    # Final clamp in-place on the output buffer.
    return out.clamp_(0.0, 1.0)

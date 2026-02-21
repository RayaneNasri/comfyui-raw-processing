import torch

def gw(img: torch.Tensor) -> torch.Tensor :
    """
    White balance image using Gray-world algorithm

    Parameters
    ----------
    img : torch.Tensor
        Image to white balance

    Returns
    -------
    img_wb : torch.Tensor
        White balanced image
    """
    
    mu, r_mu, g_mu, b_mu = torch.mean(img), torch.mean(img[..., 0]), torch.mean(img[..., 1]), torch.mean(img[..., 2])

    img_wb = img.clone()

    img_wb[..., 0] *= mu / r_mu if r_mu > 0. else 0.
    img_wb[..., 1] *= mu / g_mu if g_mu > 0. else 0.
    img_wb[..., 2] *= mu / b_mu if b_mu > 0. else 0.

    return img_wb
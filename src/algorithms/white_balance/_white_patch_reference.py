import torch

def white_patch_ref(img: torch.Tensor) -> torch.Tensor :
    """
    White balance image using White patch algorithm

    Parameters
    ----------
    img : torch.Tensor
        Image to white balance

    Returns
    -------
    img_wb : torch.Tensor
        White balanced image
    """
    r_max = torch.max(img[:, :, 0])
    g_max = torch.max(img[:, :, 1])
    b_max = torch.max(img[:, :, 2])

    img_wb = img.clone()

    img_wb[:, :, 0] /= r_max if r_max > 0. else 1.
    img_wb[:, :, 1] /= g_max if g_max > 0. else 1.
    img_wb[:, :, 2] /= b_max if b_max > 0. else 1.

    return img_wb
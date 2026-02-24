import torch

def ground_truth(img: torch.Tensor,
                 patch: torch.Tensor,
                 method: str = 'max',
                 percentil: float = 0.95
            ) -> torch.Tensor :
    """
    White balance image using Ground truth algorithm

    Parameters
    ----------
    img : torch.Tensor
        Image to white balance
    patch : torch.Tensor
        Patch of "true" white if method = 'max' else Patch of "true" gray
    method : str
        The method used 'max' or 'mean'
    percentil : float
        Between [0., 1.]
        Percentil value to consider as channel maximum
        This argument is ignored if the method argument is set to mean

    Returns
    -------
    img_wb : torch.Tensor
        White balanced image
    """

    patch_h, patch_w, patch_c = patch.shape
    
    if patch_h * patch_w == 0 :
        raise ValueError("The patch can't be empty, please use other algorithms where is not required")
    if patch_c != 3 :
        raise ValueError(f"The patch shape must be (H, W, 3), but found (H, W, {patch_c})")
    
    target = 1.

    if method == "max" :
        if (percentil < 0.) or (percentil > 1.) :
            raise ValueError(f"The percentil must be between 0 and, 1 but found {percentil}")
        if abs(percentil - 1.) < 10**(-5) :
            parameters = torch.max(patch.reshape(-1, 3), dim=0)[0]
        else :
            parameters = torch.quantile(patch.reshape(-1, 3), q=percentil, dim=0)

    elif method == "mean" :
        target = torch.mean(patch)
        parameters = torch.mean(patch.reshape(-1, 3), dim=0)
    else :
        raise ValueError(f"The method must be either `max` or `mean`, but found {method}")

    img_wb = img.clone()

    eps = 1e-6
    img_wb[..., 0] *= target / (parameters[0] + eps)
    print(f"Function {target / (parameters[0] + eps)}")
    img_wb[..., 1] *= target / (parameters[1] + eps)
    print(f"Function {target / (parameters[1] + eps)}")
    img_wb[..., 2] *= target / (parameters[2] + eps)
    print(f"Function {target / (parameters[2] + eps)}")
    
    return torch.clip(img_wb, min=0., max=1.)
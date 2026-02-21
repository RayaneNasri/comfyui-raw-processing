import torch

def ground_truth(img: torch.Tensor,
                 patch: torch.Tensor,
                 method: str = 'max'
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
    
    parameters: dict[str, float] = {"target": 1.,
                                    "red": 1.,
                                    "green": 1.,
                                    "blue": 1.
                                }
    if method == "max" :
        func = torch.max
    elif method == "mean" :
        func = torch.mean
        parameters["target"] = torch.mean(patch)
    else :
        raise ValueError(f"The method must be either `max` or `mean`, but found {method}")
    
    parameters["red"] = func(patch[..., 0])
    parameters["green"] = func(patch[..., 1])
    parameters["blue"] = func(patch[..., 2])

    img_wb = img.clone()

    if parameters["red"] > 0 :
        img_wb[..., 0] *= parameters["target"] / parameters["red"]

    if parameters["green"] > 0 :
        img_wb[..., 1] *= parameters["target"] / parameters["green"]

    if parameters["blue"] > 0 :
        img_wb[..., 2] *= parameters["target"] / parameters["blue"]
    
    return img_wb
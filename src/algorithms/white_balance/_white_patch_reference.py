import torch
import numpy as np

# FIXME: Tensor size too large with pyTorch
def white_patch_ref(img: torch.Tensor,
                    percentil: float,    
                ) -> torch.Tensor :
    """
    White balance image using White patch algorithm

    Parameters
    ----------
    img : torch.Tensor
        Image to white balance
    
    percentil : float
        Between [0., 1.]
        Percentil value to consider as channel maximum

    Returns
    -------
    img_wb : torch.Tensor
        White balanced image
    """
    if (percentil < 0.) or (percentil > 1.) :
        raise ValueError(f"The percentil must be between 0 and, 1 but found {percentil}")
    
    reshaped_image = img.reshape(-1, 3)

    if abs(percentil - 1.) < 10**(-5):
        parameters = torch.tensor(reshaped_image.max(axis=0))
    else: 
        try: 
            parameters = torch.quantile(reshaped_image, percentil, dim = 0)
            
        except RuntimeError: 
            np_reshaped_image = img.reshape(-1, 3).cpu().numpy()
            parameters = torch.tensor(np.percentile(np_reshaped_image, percentil * 100, axis=0))


    img_wb = img.clone()

    img_wb[:, :, 0] /= parameters[0] if parameters[0] > 0. else 1.
    img_wb[:, :, 1] /= parameters[1] if parameters[1] > 0. else 1.
    img_wb[:, :, 2] /= parameters[2] if parameters[2] > 0. else 1.

    return torch.clip(img_wb, min=0., max=1.)
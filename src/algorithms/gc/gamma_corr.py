import torch 

def gamma_correction(img: torch.Tensor, gamma: float= 2.2, alpha: float= 1.) -> torch.Tensor :
    """
    Performs the gamma correction of the image

    Parameters
    ----------
    img : torch.Tensor
        Image to correct
    gamma : float
        The inverse of the power to which the image is raised (default = 2.2)
    alpha : float
        A multiplier factor (default = 1)
    """
    if not isinstance(img, torch.Tensor) :
        raise TypeError("img must be a torch Tensor")
    if (gamma <= 0.):
        raise ValueError("gamma must be strictly positive")
    if (alpha < 0.):
        raise ValueError("alpha must be positive")
    
    eps = 10 ** (-6)
    return torch.clip( alpha * ( img ** (1 / (gamma + eps)) ), min=0., max=1. )
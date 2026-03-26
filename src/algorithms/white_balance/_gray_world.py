import torch


# TODO: Improve the code using a dictionary
def gray_world(img: torch.Tensor) -> torch.Tensor:
    """
    Apply the Gray-world white balance algorithm to an image.

    This algorithm assumes that the average color of a scene is gray. It calculates
    the mean of each color channel and the overall mean of the image, then scales 
    the red, green, and blue channels independently so their individual means match 
    the global mean.

    Args:
        img (torch.Tensor): The input image tensor to be white-balanced. 
            It is expected to have RGB channels in the last dimension (e.g., [H, W, 3]).

    Returns:
        torch.Tensor: The white-balanced image tensor.
    """

    mu, r_mu, g_mu, b_mu = (
        torch.mean(img),
        torch.mean(img[..., 0]),
        torch.mean(img[..., 1]),
        torch.mean(img[..., 2]),
    )

    img_wb = img.clone()

    img_wb[..., 0] *= mu / r_mu if r_mu > 0.0 else 0.0
    img_wb[..., 1] *= mu / g_mu if g_mu > 0.0 else 0.0
    img_wb[..., 2] *= mu / b_mu if b_mu > 0.0 else 0.0

    return img_wb

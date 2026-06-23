import torch


def ground_truth(
    img: torch.Tensor, patch: torch.Tensor, method: str = "max", percentil: float = 0.95
) -> torch.Tensor:
    """
    Apply white balance to an image using a Ground Truth patch algorithm.

    This function calculates scaling factors based on a provided reference patch
    that represents either "true" white or "true" gray in the scene.

    Args:
        img (torch.Tensor): The input image tensor to be white-balanced.
        patch (torch.Tensor [H, W, 3]): A reference patch from the image representing
            "true" white (if method='max') or "true" gray (if method='mean').
        method (str, optional): The method used to compute the white balance scaling
            factors. Must be either 'max' or 'mean'. Defaults to 'max'.
        percentil (float, optional): A value between [0.0, 1.0] indicating the
            percentile to consider as the channel maximum when calculating the max
            value. This argument is ignored if the method argument is set to 'mean'.
            Defaults to 0.95.

    Returns:
        torch.Tensor: The white-balanced image tensor, clipped to the range [0.0, 1.0].

    Raises:
        ValueError: If the patch is empty (width or height is 0).
        ValueError: If the patch does not have exactly 3 color channels.
        ValueError: If the percentile is not between 0.0 and 1.0 (when method='max').
        ValueError: If the method is not 'max' or 'mean'.
    """

    patch_h, patch_w, patch_c = patch.shape

    if patch_h * patch_w == 0:
        raise ValueError(
            "The patch can't be empty, please use other algorithms where is not required"
        )
    if patch_c != 3:
        raise ValueError(
            f"The patch shape must be (H, W, 3), but found (H, W, {patch_c})"
        )

    target = 1.0

    if method == "max":
        if (percentil < 0.0) or (percentil > 1.0):
            raise ValueError(
                f"The percentil must be between 0 and, 1 but found {percentil}"
            )
        if abs(percentil - 1.0) < 10 ** (-5):
            parameters = torch.max(patch.reshape(-1, 3), dim=0)[0]
        else:
            parameters = torch.quantile(patch.reshape(-1, 3), q=percentil, dim=0)

    elif method == "mean":
        target = torch.mean(patch)
        parameters = torch.mean(patch.reshape(-1, 3), dim=0)
    else:
        raise ValueError(
            f"The method must be either `max` or `mean`, but found {method}"
        )

    img_wb = img.clone()

    eps = 1e-6
    img_wb[..., 0] *= target / (parameters[0] + eps)
    print(f"Function {target / (parameters[0] + eps)}")
    img_wb[..., 1] *= target / (parameters[1] + eps)
    print(f"Function {target / (parameters[1] + eps)}")
    img_wb[..., 2] *= target / (parameters[2] + eps)
    print(f"Function {target / (parameters[2] + eps)}")

    return torch.clip(img_wb, min=0.0, max=1.0)

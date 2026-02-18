import torch


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    if den.item() == 0:
        return torch.zeros((), dtype=num.dtype, device=num.device)
    return num / den


def malvar_he_cutler_demosaicing(
    raw_image: torch.Tensor,
    dx: int = 0,
    dy: int = 0,
) -> torch.Tensor:
    if raw_image.ndim != 2:
        raise ValueError("raw_image must be a 2D tensor of shape (H, W)")
    if dx not in (0, 1) or dy not in (0, 1):
        raise ValueError("dx and dy must be 0 or 1")

    height, width = raw_image.shape

    red_x, red_y = dx, dy
    blue_x, blue_y = 1 - red_x, 1 - red_y

    output = torch.empty((height, width, 3), dtype=raw_image.dtype, device=raw_image.device)

    for y in range(height):
        for x in range(width):
            neighbors = torch.zeros((5, 5), dtype=raw_image.dtype, device=raw_image.device)
            neighbor_presence = torch.zeros((5, 5), dtype=raw_image.dtype, device=raw_image.device)

            for ny in range(-2, 3):
                for nx in range(-2, 3):
                    sx = x + nx
                    sy = y + ny
                    if 0 <= sx < width and 0 <= sy < height:
                        neighbors[nx + 2, ny + 2] = raw_image[sy, sx]
                        neighbor_presence[nx + 2, ny + 2] = 1

            center = neighbors[2, 2]

            is_red = (x & 1) == red_x and (y & 1) == red_y
            is_blue = (x & 1) == blue_x and (y & 1) == blue_y

            if is_red:
                out_r = raw_image[y, x]

                green_num = (
                    2 * (neighbors[2, 1] + neighbors[1, 2] + neighbors[3, 2] + neighbors[2, 3])
                    + (
                        neighbor_presence[0, 2]
                        + neighbor_presence[4, 2]
                        + neighbor_presence[2, 0]
                        + neighbor_presence[2, 4]
                    )
                    * center
                    - neighbors[0, 2]
                    - neighbors[4, 2]
                    - neighbors[2, 0]
                    - neighbors[2, 4]
                )
                green_den = 2 * (
                    neighbor_presence[2, 1]
                    + neighbor_presence[1, 2]
                    + neighbor_presence[3, 2]
                    + neighbor_presence[2, 3]
                )
                out_g = _safe_div(green_num, green_den)

                blue_num = 4 * (neighbors[1, 1] + neighbors[3, 1] + neighbors[1, 3] + neighbors[3, 3]) + 3 * (
                    (
                        neighbor_presence[0, 2]
                        + neighbor_presence[4, 2]
                        + neighbor_presence[2, 0]
                        + neighbor_presence[2, 4]
                    )
                    * center
                    - neighbors[0, 2]
                    - neighbors[4, 2]
                    - neighbors[2, 0]
                    - neighbors[2, 4]
                )
                blue_den = 4 * (
                    neighbor_presence[1, 1]
                    + neighbor_presence[3, 1]
                    + neighbor_presence[1, 3]
                    + neighbor_presence[3, 3]
                )
                out_b = _safe_div(blue_num, blue_den)

            elif is_blue:
                out_b = raw_image[y, x]

                green_num = (
                    2 * (neighbors[2, 1] + neighbors[1, 2] + neighbors[3, 2] + neighbors[2, 3])
                    + (
                        neighbor_presence[0, 2]
                        + neighbor_presence[4, 2]
                        + neighbor_presence[2, 0]
                        + neighbor_presence[2, 4]
                    )
                    * center
                    - neighbors[0, 2]
                    - neighbors[4, 2]
                    - neighbors[2, 0]
                    - neighbors[2, 4]
                )
                green_den = 2 * (
                    neighbor_presence[2, 1]
                    + neighbor_presence[1, 2]
                    + neighbor_presence[3, 2]
                    + neighbor_presence[2, 3]
                )
                out_g = _safe_div(green_num, green_den)

                red_num = 4 * (neighbors[1, 1] + neighbors[3, 1] + neighbors[1, 3] + neighbors[3, 3]) + 3 * (
                    (
                        neighbor_presence[0, 2]
                        + neighbor_presence[4, 2]
                        + neighbor_presence[2, 0]
                        + neighbor_presence[2, 4]
                    )
                    * center
                    - neighbors[0, 2]
                    - neighbors[4, 2]
                    - neighbors[2, 0]
                    - neighbors[2, 4]
                )
                red_den = 4 * (
                    neighbor_presence[1, 1]
                    + neighbor_presence[3, 1]
                    + neighbor_presence[1, 3]
                    + neighbor_presence[3, 3]
                )
                out_r = _safe_div(red_num, red_den)

            else:
                out_g = raw_image[y, x]

                if (y & 1) == red_y:
                    red_num = (
                        8 * (neighbors[1, 2] + neighbors[3, 2])
                        + (
                            2
                            * (
                                neighbor_presence[1, 1]
                                + neighbor_presence[3, 1]
                                + neighbor_presence[0, 2]
                                + neighbor_presence[4, 2]
                                + neighbor_presence[1, 3]
                                + neighbor_presence[3, 3]
                            )
                            - neighbor_presence[2, 0]
                            - neighbor_presence[2, 4]
                        )
                        * center
                        - 2
                        * (
                            neighbors[1, 1]
                            + neighbors[3, 1]
                            + neighbors[0, 2]
                            + neighbors[4, 2]
                            + neighbors[1, 3]
                            + neighbors[3, 3]
                        )
                        + neighbors[2, 0]
                        + neighbors[2, 4]
                    )
                    red_den = 8 * (neighbor_presence[1, 2] + neighbor_presence[3, 2])
                    out_r = _safe_div(red_num, red_den)

                    blue_num = (
                        8 * (neighbors[2, 1] + neighbors[2, 3])
                        + (
                            2
                            * (
                                neighbor_presence[1, 1]
                                + neighbor_presence[3, 1]
                                + neighbor_presence[2, 0]
                                + neighbor_presence[2, 4]
                                + neighbor_presence[1, 3]
                                + neighbor_presence[3, 3]
                            )
                            - neighbor_presence[0, 2]
                            - neighbor_presence[4, 2]
                        )
                        * center
                        - 2
                        * (
                            neighbors[1, 1]
                            + neighbors[3, 1]
                            + neighbors[2, 0]
                            + neighbors[2, 4]
                            + neighbors[1, 3]
                            + neighbors[3, 3]
                        )
                        + neighbors[0, 2]
                        + neighbors[4, 2]
                    )
                    blue_den = 8 * (neighbor_presence[2, 1] + neighbor_presence[2, 3])
                    out_b = _safe_div(blue_num, blue_den)
                else:
                    red_num = (
                        8 * (neighbors[2, 1] + neighbors[2, 3])
                        + (
                            2
                            * (
                                neighbor_presence[1, 1]
                                + neighbor_presence[3, 1]
                                + neighbor_presence[2, 0]
                                + neighbor_presence[2, 4]
                                + neighbor_presence[1, 3]
                                + neighbor_presence[3, 3]
                            )
                            - neighbor_presence[0, 2]
                            - neighbor_presence[4, 2]
                        )
                        * center
                        - 2
                        * (
                            neighbors[1, 1]
                            + neighbors[3, 1]
                            + neighbors[2, 0]
                            + neighbors[2, 4]
                            + neighbors[1, 3]
                            + neighbors[3, 3]
                        )
                        + neighbors[0, 2]
                        + neighbors[4, 2]
                    )
                    red_den = 8 * (neighbor_presence[2, 1] + neighbor_presence[2, 3])
                    out_r = _safe_div(red_num, red_den)

                    blue_num = (
                        8 * (neighbors[1, 2] + neighbors[3, 2])
                        + (
                            2
                            * (
                                neighbor_presence[1, 1]
                                + neighbor_presence[3, 1]
                                + neighbor_presence[0, 2]
                                + neighbor_presence[4, 2]
                                + neighbor_presence[1, 3]
                                + neighbor_presence[3, 3]
                            )
                            - neighbor_presence[2, 0]
                            - neighbor_presence[2, 4]
                        )
                        * center
                        - 2
                        * (
                            neighbors[1, 1]
                            + neighbors[3, 1]
                            + neighbors[0, 2]
                            + neighbors[4, 2]
                            + neighbors[1, 3]
                            + neighbors[3, 3]
                        )
                        + neighbors[2, 0]
                        + neighbors[2, 4]
                    )
                    blue_den = 8 * (neighbor_presence[1, 2] + neighbor_presence[3, 2])
                    out_b = _safe_div(blue_num, blue_den)

            output[y, x, 0] = out_r
            output[y, x, 1] = out_g
            output[y, x, 2] = out_b

    return output

import numpy as np


def bilinear_demosaicing(rgb_image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    dy, dx : # offset on the raw, tells you the location of the first pixel sampled in red (dy is equal to 0 or 1, and dx is equal 0 or 1)
    """
    # Bayer CFA with (dy,dx) the location of the first pixel sampled in red
    # pixels in rgb_image[dy::2, dx::2] are sampled in red
    # those in rgb_image[1-dy::2, 1-dx::2] in blue
    # those in rgb_image[dy::2, 1-dx::2] and in rgb_image[1-dy::2, dx::2] in green.
    # Example of Bayer CFA with (dy,dx) = (0,0):
    # R G R G R
    # G B G B G
    # R G R G R

    height, width, _ = rgb_image.shape

    # Zero boundary condition :
    # Frame the image with zeros (Add 1 row and 1 column of zeros before and after the height/width dimensions; nothing on the RGB channels.)
    # padded_rgb_image.shape = (height+2, width+2, 3)
    padded_rgb_image = np.pad(
        rgb_image,
        pad_width=((1, 1), (1, 1), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    demosaicing_image = padded_rgb_image.copy()

    # nbpix_C1_C2 : if a pixel has a colour C1, it has "nbpix_C1_C2" pixels of colour C2 as direct neighbours
    # This number depends on the location of the pixel (corner, edge, center) and of (dy,dx)
    nbpix_r_g = 0
    nbpix_r_b = 0
    nbpix_g_r = 0
    nbpix_g_b = 0
    nbpix_b_r = 0
    nbpix_b_g = 0

    for i in range(1, height + 1):
        for j in range(1, width + 1):

            ### Values of nbpix_C1_C2
            # corners
            if (
                (i == 1 and j == 1)
                or (i == 1 and j == width)
                or (i == height and j == 1)
                or (i == height and j == width)
            ):
                if ((i - 1) % 2 == 1 - dy and (j - 1) % 2 == dx) or (
                    (i - 1) % 2 == dy and (j - 1) % 2 == 1 - dx
                ):  # green pixel
                    nbpix_g_r = 1  # if the pixel is green, it has 1 red pixel as a direct neighbour
                    nbpix_g_b = 1
                else:  # red or blue pixel
                    nbpix_r_b = 1
                    nbpix_b_r = 1
                    nbpix_r_g = 2  # if the pixel is red, it has 2 green pixels as direct neighbours
                    nbpix_b_g = 2

            # edge (without corners)
            elif i == 1 or j == 1 or i == height or j == width:
                if ((i - 1) % 2 == 1 - dy and (j - 1) % 2 == dx) or (
                    (i - 1) % 2 == dy and (j - 1) % 2 == 1 - dx
                ):  # green pixel
                    if (
                        (
                            i == 1 and dy == 0
                        )  # if the pixels of the first row are red and green
                        or (
                            j == 1 and dx == 0
                        )  # if the pixels of the first column are red and green
                        or (
                            i == height
                            and (
                                (dy == 0 and height % 2 == 1)
                                or (dy == 1 and height % 2 == 0)
                            )
                        )  # if the pixels of the last row are red and green
                        or (
                            j == width
                            and (
                                (dx == 0 and width % 2 == 1)
                                or (dx == 1 and width % 2 == 0)
                            )
                        )  # if the pixels of the last column are red and green
                    ):
                        nbpix_g_r = 2
                        nbpix_g_b = 1
                    else:  # if the edge in question is blue and green.
                        nbpix_g_r = 1
                        nbpix_g_b = 2
                else:  # red or blue pixel
                    nbpix_r_b = 2
                    nbpix_b_r = 2
                    nbpix_r_g = 3
                    nbpix_b_g = 3

            # center of the image
            else:
                nbpix_r_g = 4
                nbpix_r_b = 4
                nbpix_g_r = (
                    2  # if the pixel is green, it has 2 red pixels as direct neighbours
                )
                nbpix_g_b = 2
                nbpix_b_r = 4
                nbpix_b_g = 4

            ### Values of demosaicing_image[i, j, .]
            if (i - 1) % 2 == dy and (
                j - 1
            ) % 2 == dx:  # red pixel (we can't use "rgb_image[i, j, 0] != 0" because a pixel can have (R=0, G=0, B=0))
                # green
                demosaicing_image[i, j, 1] = (
                    padded_rgb_image[i - 1, j, 1]
                    + padded_rgb_image[i + 1, j, 1]
                    + padded_rgb_image[i, j - 1, 1]
                    + padded_rgb_image[i, j + 1, 1]
                ) / nbpix_r_g
                # blue
                demosaicing_image[i, j, 2] = (
                    padded_rgb_image[i - 1, j - 1, 2]
                    + padded_rgb_image[i - 1, j + 1, 2]
                    + padded_rgb_image[i + 1, j - 1, 2]
                    + padded_rgb_image[i + 1, j + 1, 2]
                ) / nbpix_r_b
            elif ((i - 1) % 2 == 1 - dy and (j - 1) % 2 == dx) or (
                (i - 1) % 2 == dy and (j - 1) % 2 == 1 - dx
            ):  # green pixel
                # red
                demosaicing_image[i, j, 0] = (
                    padded_rgb_image[i - 1, j, 0]
                    + padded_rgb_image[i + 1, j, 0]
                    + padded_rgb_image[i, j - 1, 0]
                    + padded_rgb_image[i, j + 1, 0]
                ) / nbpix_g_r
                # blue
                demosaicing_image[i, j, 2] = (
                    padded_rgb_image[i - 1, j, 2]
                    + padded_rgb_image[i + 1, j, 2]
                    + padded_rgb_image[i, j - 1, 2]
                    + padded_rgb_image[i, j + 1, 2]
                ) / nbpix_g_b
            else:  # blue pixel : (i-1)%2 == 1-dy and (j-1)%2 == 1-dx
                # red
                demosaicing_image[i, j, 0] = (
                    padded_rgb_image[i - 1, j - 1, 0]
                    + padded_rgb_image[i - 1, j + 1, 0]
                    + padded_rgb_image[i + 1, j - 1, 0]
                    + padded_rgb_image[i + 1, j + 1, 0]
                ) / nbpix_b_r
                # green
                demosaicing_image[i, j, 1] = (
                    padded_rgb_image[i - 1, j, 1]
                    + padded_rgb_image[i + 1, j, 1]
                    + padded_rgb_image[i, j - 1, 1]
                    + padded_rgb_image[i, j + 1, 1]
                ) / nbpix_b_g

    # return th demosaicing_image wthout the frame of zeros
    return demosaicing_image[1:-1, 1:-1, :]

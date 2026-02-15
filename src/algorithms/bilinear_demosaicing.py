import numpy as np  

def mono_to_rgb(normalized_image: np.ndarray, bayer_pattern: np.ndarray)-> np.ndarray:
    """
    the docstring ...
    """
    height, width = normalized_image.shape
    # 2. Initialize the output array (H, W, 3) with zeros
    rgb_image = np.zeros((height, width, 3), dtype=normalized_image.dtype)
    # 3. Handle the 4-color Bayer case (RGGB usually has indices 0, 1, 2, 3)
    # We map index 3 (Green2) to index 1 (Green) so it fits in a 3-channel RGB image.
    # We use a copy to avoid modifying the original input array.
    channel_indices = bayer_pattern.copy()
    channel_indices[channel_indices == 3] = 1
    # 4. Use Advanced Indexing (Vectorization)
    # We create coordinate grids to assign all values at once.
    # rows[:, None] creates a column vector (H, 1)
    # np.arange(width) creates a row vector (1, W)
    # NumPy broadcasts these against 'channel_indices' (H, W) to select exact targets.
    rows = np.arange(height)[:, None]
    cols = np.arange(width)
    # This single line replaces the nested for-loops
    rgb_image[rows, cols, channel_indices] = normalized_image
    return rgb_image

def bilinear_demosaicing(rgb_image: np.ndarray, dx:int, dy:int)-> np.ndarray:
    """
    dy, dx : # offset on the raw, tells you the location of the first pixel sampled in red
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
    padded_rgb_image = np.pad(rgb_image, pad_width=((1,1), (1,1), (0,0)), mode='constant', constant_values=0)
    
    demosaicing_image = padded_rgb_image.copy()


    for i in range(1,height+1):
        for j in range(1,width+1):
            if (i-1)%2 == dy and (j-1)%2 == dx: # red pixel (we can't use "rgb_image[i, j, 0] != 0" because a pixel can have (R=0, G=0, B=0))
                # green
                demosaicing_image[i, j, 1] = (padded_rgb_image[i-1, j, 1] + padded_rgb_image[i+1, j, 1] + padded_rgb_image[i, j-1, 1] + padded_rgb_image[i, j+1, 1]) / 4
                # blue
                demosaicing_image[i, j, 2] = (padded_rgb_image[i-1, j-1, 2] + padded_rgb_image[i-1, j+1, 2] + padded_rgb_image[i+1, j-1, 2] + padded_rgb_image[i+1, j+1, 2]) / 4
            if ((i-1)%2 == 1-dy and (j-1)%2 == dx) or ((i-1)%2 == dy and (j-1)%2 == 1-dx): # green pixel
                # red (It is divided by 2 and not 4 because only two of the four pixels are red in the Bayer CFA)
                demosaicing_image[i, j, 0] = (padded_rgb_image[i-1, j, 0] + padded_rgb_image[i+1, j, 0] + padded_rgb_image[i, j-1, 0] + padded_rgb_image[i, j+1, 0]) / 2
                # blue (idem)
                demosaicing_image[i, j, 2] = (padded_rgb_image[i-1, j, 2] + padded_rgb_image[i+1, j, 2] + padded_rgb_image[i, j-1, 2] + padded_rgb_image[i, j+1, 2]) / 2
            if (i-1)%2 == 1-dy and (j-1)%2 == 1-dx: # blue pixel
                # red
                demosaicing_image[i, j, 0] = (padded_rgb_image[i-1, j-1, 0] + padded_rgb_image[i-1, j+1, 0] + padded_rgb_image[i+1, j-1, 0] + padded_rgb_image[i+1, j+1, 0]) / 4
                # green
                demosaicing_image[i, j, 1] = (padded_rgb_image[i-1, j, 1] + padded_rgb_image[i+1, j, 1] + padded_rgb_image[i, j-1, 1] + padded_rgb_image[i, j+1, 1]) / 4
    
    # return th demosaicing_image wthout the frame of zeros
    return demosaicing_image[1:-1, 1:-1, :]

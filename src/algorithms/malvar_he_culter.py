import numpy as np  

from numpy import ndarray

def malvar_he_culter(rgb_image: ndarray, bayer_pattern: ndarray) -> ndarray: 
    """ Performs the algorithm of Malvar He Culter for bilinear demoisaicing """
    
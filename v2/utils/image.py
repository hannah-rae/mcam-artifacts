
import numpy as np


def arr_to_rgba(arr):
    return np.dstack((
        (np.flipud(arr)*256).astype(np.uint8),
        np.full(arr.shape[0:2], 255, dtype=np.uint8)
    )).view(dtype=np.uint32).reshape(arr.shape[0:2])

def arr_to_img(arr):
    return (arr*256).astype(np.uint8)
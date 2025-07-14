import imageio.v3 as imageio
import numpy as np


def read_bayer_image(path: str):
    raw = imageio.imread(path)
    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]
    return np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))


def read_rgb_image(path: str) -> np.ndarray:
    return imageio.imread(path)


def read_numpy_feature(path: str) -> np.ndarray:
    return np.load(path)

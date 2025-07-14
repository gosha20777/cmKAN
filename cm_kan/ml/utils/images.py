import cv2 as cv
import numpy as np
import colour
# np.set_printoptions(threshold=sys.maxsize)


def normalize(image: np.ndarray):
    """input image normalized [0,255]"""
    return image / 255.0


def denormalize(image: np.ndarray):
    """input image normalized [0,1]"""
    return image * 255.0


def validate(image: np.ndarray):
    image[image > 1.0] = 1.0
    image[image < 0.0] = 0.0
    return image


def srgb_to_rgb(image: np.ndarray):
    """srgb input image normalized [0,1]"""
    sRGB = colour.RGB_COLOURSPACES["sRGB"]
    return sRGB.cctf_decoding(image)


def rgb_to_srgb(image: np.ndarray):
    """xyz input image"""
    sRGB = colour.RGB_COLOURSPACES["sRGB"]
    return sRGB.cctf_encoding(image)


def srgb_to_xyz(image: np.ndarray):
    """srgb input image normalized [0,1]"""
    return colour.sRGB_to_XYZ(image)


def xyz_to_srgb(image: np.ndarray):
    """xyz input image"""
    return colour.XYZ_to_sRGB(image)


# PIL images


def rgb(image: np.ndarray) -> np.ndarray:
    """Converts srgb image [0,255] to linear rgb image [0,1]"""
    img = np.array(image)
    img_norm = normalize(img)
    return srgb_to_rgb(img_norm)


def srgb(image: np.ndarray) -> np.ndarray:
    """Converts linear rgb image [0,1] to srgb image [0,255]"""
    img_valid = validate(image)
    img_norm = rgb_to_srgb(img_valid)
    return denormalize(img_norm)


def read(path: str) -> np.ndarray:
    """Reads srgb image [0,255]"""
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def write(path: str, image: np.ndarray):
    """Reads srgb image [0,255]"""
    image = cv.cvtColor(image.astype(np.uint8), cv.COLOR_RGB2BGR)
    cv.imwrite(str(path), image)

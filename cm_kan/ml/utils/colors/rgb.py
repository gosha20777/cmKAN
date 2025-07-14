import torch


def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to BGR.

    .. image:: _static/img/rgb_to_bgr.png

    Args:
        image: RGB Image to be converted to BGRof of shape :math:`(*,3,H,W)`.

    Returns:
        BGR version of the image with shape of shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_bgr(input) # 2x3x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W).Got {image.shape}"
        )

    return bgr_to_rgb(image)


def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a BGR image to RGB.

    Args:
        image: BGR Image to be converted to BGR of shape :math:`(*,3,H,W)`.

    Returns:
        RGB version of the image with shape of shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = bgr_to_rgb(input) # 2x3x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W).Got {image.shape}"
        )

    # flip image channels
    out: torch.Tensor = image.flip(-3)
    return out


def rgb_to_linear_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an sRGB image to linear RGB. Used in colorspace conversions.

    .. image:: _static/img/rgb_to_linear_rgb.png

    Args:
        image: sRGB Image to be converted to linear RGB of shape :math:`(*,3,H,W)`.

    Returns:
        linear RGB version of the image with shape of :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_linear_rgb(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W).Got {image.shape}"
        )

    lin_rgb: torch.Tensor = torch.where(
        image > 0.04045, torch.pow(((image + 0.055) / 1.055), 2.4), image / 12.92
    )

    return lin_rgb


def linear_rgb_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a linear RGB image to sRGB. Used in colorspace conversions.

    Args:
        image: linear RGB Image to be converted to sRGB of shape :math:`(*,3,H,W)`.

    Returns:
        sRGB version of the image with shape of shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = linear_rgb_to_rgb(input) # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W).Got {image.shape}"
        )

    threshold = 0.0031308
    rgb: torch.Tensor = torch.where(
        image > threshold,
        1.055 * torch.pow(image.clamp(min=threshold), 1 / 2.4) - 0.055,
        12.92 * image,
    )

    return rgb


class BgrToRgb(torch.nn.Module):
    r"""Convert image from BGR to RGB.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = BgrToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return bgr_to_rgb(image)


class RgbToBgr(torch.nn.Module):
    r"""Convert an image from RGB to BGR.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        BGR version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> bgr = RgbToBgr()
        >>> output = bgr(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_bgr(image)


class RgbToLinearRgb(torch.nn.Module):
    r"""Convert an image from sRGB to linear RGB.

    Reverses the gamma correction of sRGB to get linear RGB values for colorspace conversions.
    The image data is assumed to be in the range of :math:`[0, 1]`

    Returns:
        Linear RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb_lin = RgbToLinearRgb()
        >>> output = rgb_lin(input)  # 2x3x4x5

    References:
        [1] https://stackoverflow.com/questions/35952564/convert-rgb-to-srgb

        [2] https://www.cambridgeincolour.com/tutorials/gamma-correction.htm

        [3] https://en.wikipedia.org/wiki/SRGB
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_linear_rgb(image)


class LinearRgbToRgb(torch.nn.Module):
    r"""Convert a linear RGB image to sRGB.

    Applies gamma correction to linear RGB values, at the end of colorspace conversions, to get sRGB.

    Returns:
        sRGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> srgb = LinearRgbToRgb()
        >>> output = srgb(input)  # 2x3x4x5

    References:
        [1] https://stackoverflow.com/questions/35952564/convert-rgb-to-srgb

        [2] https://www.cambridgeincolour.com/tutorials/gamma-correction.htm

        [3] https://en.wikipedia.org/wiki/SRGB
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return linear_rgb_to_rgb(image)

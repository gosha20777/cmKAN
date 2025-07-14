import torch
from typing import List
from functools import reduce
from torchmetrics import Metric
from ..utils.colors import rgb_to_lab


def _tsplit(a: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Reshape input image [b, c, h, w] or color feature [b, f, c] to [L, a, b] format.

    Parameters
    ----------
    a
        Input tensor :math:`a` to split.
    dtype
        :class:`numpy.dtype` to use for initial conversion

    Returns
    -------
    :class:`torch.Tensor`
        Tensor of tensors.

    Examples
    --------
    >>> a = torch.Tensor([0, 0, 0])
    >>> tsplit(a)
    torch.Tensor([ 0.,  0.,  0.])
    >>> a = torch.Tensor(
    ...     [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]
    ... )
    >>> tsplit(a)
    torch.Tensor([[ 0.,  1.,  2.,  3.,  4.,  5.],
           [ 0.,  1.,  2.,  3.,  4.,  5.],
           [ 0.,  1.,  2.,  3.,  4.,  5.]])
    >>> a = torch.Tensor(
    ...     [
    ...         [
    ...             [0, 0, 0],
    ...             [1, 1, 1],
    ...             [2, 2, 2],
    ...             [3, 3, 3],
    ...             [4, 4, 4],
    ...             [5, 5, 5],
    ...         ]
    ...     ]
    ... )
    >>> tsplit(a)
    torch.Tensor([[[ 0.,  1.,  2.,  3.,  4.,  5.]],
    <BLANKLINE>
           [[ 0.,  1.,  2.,  3.,  4.,  5.]],
    <BLANKLINE>
           [[ 0.,  1.,  2.,  3.,  4.,  5.]]])
    """

    a = a.to(dtype)

    if a.ndim <= 3:
        return torch.transpose(a, a.ndim - 1, 0)

    return torch.transpose(a, 1, 0)


def _select(
    conditions: List[torch.Tensor], values: List[torch.Tensor], default: int = 0
) -> torch.Tensor:
    """
    Return a torch.Tensor drawn from elements in choicelist, depending on conditions.

    Parameters
    ----------
    conditions : list of bool tensors
        The list of conditions which determine from which array in `choicelist`
        the output elements are taken. When multiple conditions are satisfied,
        the first one encountered in `condlist` is used.
    values : list of tensors
        The list of arrays from which the output elements are taken. It has
        to be of the same length as `condlist`.
    default : scalar, optional
        The element inserted in `output` when all conditions evaluate to False.

    Returns
    -------
    output : torch.Tensor
        The output at position m is the m-th element of the array in
        `choicelist` where the m-th element of the corresponding array in
        `condlist` is True.

    See Also
    --------
    torch.where : Return elements from one of two arrays depending on condition.
    take, choose, compress, diag, diagonal

    Examples
    --------

    >>> x = torch.arange(6)
    >>> condlist = [x<3, x>3]
    >>> choicelist = [x, x**2]
    >>> select(condlist, choicelist, 42)
    torch.Tensor([ 0,  1,  2, 42, 16, 25])
    """
    zipped = reversed(list(zip(conditions, values)))
    return reduce(lambda o, a: torch.where(*a, o), zipped, default)


def _delta_E_CIE2000(
    Lab_1: torch.Tensor, Lab_2: torch.Tensor, textiles: bool = False
) -> torch.Tensor:
    """
    Return the difference :math:`\\Delta E_{00}` between two given
    *CIE L\\*a\\*b\\** colourspace tensors using *CIE 2000* recommendation.

    Parameters
    ----------
    Lab_1
        *CIE L\\*a\\*b\\** colourspace tensor 1.
    Lab_2
        *CIE L\\*a\\*b\\** colourspace tensor 2.
    textiles
        Textiles application specific parametric factors.
        :math:`k_L=2,\\ k_C=k_H=1` weights are used instead of
        :math:`k_L=k_C=k_H=1`.

    Returns
    -------
    :class:`torch.Tensor`
        Colour difference :math:`\\Delta E_{00}`.

    Notes
    -----
    +------------+-----------------------+
    | **Domain** |        **Scale**      |
    +============+=======================+
    | ``Lab_1``  | ``L_1`` : [0, 100]    |
    |            |                       |
    |            | ``a_1`` : [-100, 100] |
    |            |                       |
    |            | ``b_1`` : [-100, 100] |
    +------------+-----------------------+
    | ``Lab_2``  | ``L_2`` : [0, 100]    |
    |            |                       |
    |            | ``a_2`` : [-100, 100] |
    |            |                       |
    |            | ``b_2`` : [-100, 100] |
    +------------+-----------------------+

    -   Parametric factors :math:`k_L=k_C=k_H=1` weights under
        *reference conditions*:

        -   Illumination: D65 source
        -   Illuminance: 1000 lx
        -   Observer: Normal colour vision
        -   Background field: Uniform, neutral gray with :math:`L^*=50`
        -   Viewing mode: Object
        -   Sample size: Greater than 4 degrees
        -   Sample separation: Direct edge contact
        -   Sample colour-difference magnitude: Lower than 5.0
            :math:`\\Delta E_{00}`
        -   Sample structure: Homogeneous (without texture)

    References
    ----------
    :cite:`Melgosa2013b`, :cite:`Sharma2005b`

    Examples
    --------
    >>> Lab_1 = torch.Tensor([100.00000000, 21.57210357, 272.22819350])
    >>> Lab_2 = torch.Tensor([100.00000000, 426.67945353, 72.39590835])
    >>> delta_E_CIE2000(Lab_1, Lab_2)  # doctest: +ELLIPSIS
    94.0356490...
    >>> Lab_2 = torch.Tensor([50.00000000, 426.67945353, 72.39590835])
    >>> delta_E_CIE2000(Lab_1, Lab_2)  # doctest: +ELLIPSIS
    100.8779470...
    >>> delta_E_CIE2000(Lab_1, Lab_2, textiles=True)  # doctest: +ELLIPSIS
    95.7920535...
    """

    L_1, a_1, b_1 = _tsplit(Lab_1)
    L_2, a_2, b_2 = _tsplit(Lab_2)

    k_L = 2 if textiles else 1
    k_C = 1
    k_H = 1

    C_1_ab = torch.hypot(a_1, b_1)
    C_2_ab = torch.hypot(a_2, b_2)

    C_bar_ab = (C_1_ab + C_2_ab) / 2
    C_bar_ab_7 = C_bar_ab**7

    G = 0.5 * (1 - torch.sqrt(C_bar_ab_7 / (C_bar_ab_7 + 25**7)))

    a_p_1 = (1 + G) * a_1
    a_p_2 = (1 + G) * a_2

    C_p_1 = torch.hypot(a_p_1, b_1)
    C_p_2 = torch.hypot(a_p_2, b_2)

    h_p_1 = torch.where(
        torch.logical_and(b_1 == 0, a_p_1 == 0),
        0,
        torch.rad2deg(torch.arctan2(b_1, a_p_1)) % 360,
    )
    h_p_2 = torch.where(
        torch.logical_and(b_2 == 0, a_p_2 == 0),
        0,
        torch.rad2deg(torch.arctan2(b_2, a_p_2)) % 360,
    )

    delta_L_p = L_2 - L_1

    delta_C_p = C_p_2 - C_p_1

    h_p_2_s_1 = h_p_2 - h_p_1
    C_p_1_m_2 = C_p_1 * C_p_2
    delta_h_p = _select(
        [
            C_p_1_m_2 == 0,
            torch.abs(h_p_2_s_1) <= 180,
            h_p_2_s_1 > 180,
            h_p_2_s_1 < -180,
        ],
        [
            0,
            h_p_2_s_1,
            h_p_2_s_1 - 360,
            h_p_2_s_1 + 360,
        ],
    )

    delta_H_p = 2 * torch.sqrt(C_p_1_m_2) * torch.sin(torch.deg2rad(delta_h_p / 2))

    L_bar_p = (L_1 + L_2) / 2

    C_bar_p = (C_p_1 + C_p_2) / 2

    a_h_p_1_s_2 = torch.abs(h_p_1 - h_p_2)
    h_p_1_a_2 = h_p_1 + h_p_2
    h_bar_p = _select(
        [
            C_p_1_m_2 == 0,
            a_h_p_1_s_2 <= 180,
            torch.logical_and(a_h_p_1_s_2 > 180, h_p_1_a_2 < 360),
            torch.logical_and(a_h_p_1_s_2 > 180, h_p_1_a_2 >= 360),
        ],
        [
            h_p_1_a_2,
            h_p_1_a_2 / 2,
            (h_p_1_a_2 + 360) / 2,
            (h_p_1_a_2 - 360) / 2,
        ],
    )

    T = (
        1
        - 0.17 * torch.cos(torch.deg2rad(h_bar_p - 30))
        + 0.24 * torch.cos(torch.deg2rad(2 * h_bar_p))
        + 0.32 * torch.cos(torch.deg2rad(3 * h_bar_p + 6))
        - 0.20 * torch.cos(torch.deg2rad(4 * h_bar_p - 63))
    )

    delta_theta = 30 * torch.exp(-(((h_bar_p - 275) / 25) ** 2))

    C_bar_p_7 = C_bar_p**7
    R_C = 2 * torch.sqrt(C_bar_p_7 / (C_bar_p_7 + 25**7))

    L_bar_p_2 = (L_bar_p - 50) ** 2
    S_L = 1 + ((0.015 * L_bar_p_2) / torch.sqrt(20 + L_bar_p_2))

    S_C = 1 + 0.045 * C_bar_p

    S_H = 1 + 0.015 * C_bar_p * T

    R_T = -torch.sin(torch.deg2rad(2 * delta_theta)) * R_C

    d_E = torch.sqrt(
        (delta_L_p / (k_L * S_L)) ** 2
        + (delta_C_p / (k_C * S_C)) ** 2
        + (delta_H_p / (k_H * S_H)) ** 2
        + R_T * (delta_C_p / (k_C * S_C)) * (delta_H_p / (k_H * S_H))
    )

    return d_E


class DeltaE(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "correct",
            default=torch.tensor(0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        preds = rgb_to_lab(preds)
        target = rgb_to_lab(target)
        de = torch.mean(_delta_E_CIE2000(preds, target), dtype=torch.float64)

        self.correct += de
        self.total += 1

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total

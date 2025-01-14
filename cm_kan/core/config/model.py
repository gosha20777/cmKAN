from pydantic import BaseModel
from enum import Enum
from typing import List, Union


class ModelType(str, Enum):
    simple_kan = 'simple_kan'
    conv_kan = 'conv_kan'
    mw_isp = 'mw_isp'
    sep_kan = 'sep_kan'
    conv_sep_kan = 'conv_sep_kan'
    sms = 'sms'
    sep_kan_2d = 'sep_kan_2d'
    

class SMSModelParams(BaseModel):
    pass

class ConvSepKanModelParams(BaseModel):
    in_dims: List[int]
    out_dims: List[int]
    kernel_sizes: List[int]
    grid_size: int
    spline_order: int
    residual_std: float
    grid_range: List[float]


class SimpleKanModelParams(BaseModel):
    in_dims: List[int]
    out_dims: List[int]
    grid_size: int
    spline_order: int
    residual_std: float
    grid_range: List[float]


class ConvKanModelParams(BaseModel):
    in_channels: int
    out_channels: int
    grid_size: int
    spline_order: int
    residual_std: float
    grid_range: List[float]


class MwIspModelParams(BaseModel):
    in_channels: int
    out_channels: int


class Model(BaseModel):
    type: ModelType
    params: Union[
        ConvSepKanModelParams,
        SimpleKanModelParams,
        ConvKanModelParams,
        MwIspModelParams,
        SMSModelParams
    ]

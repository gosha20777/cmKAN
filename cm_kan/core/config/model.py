from pydantic import BaseModel
from enum import Enum
from typing import List, Union


class ModelType(str, Enum):
    cm_kan = 'cm_kan'
    

class CmKanModelParams(BaseModel):
    in_dims: List[int]
    out_dims: List[int]
    kernel_sizes: List[int]
    grid_size: int
    spline_order: int
    residual_std: float
    grid_range: List[float]


class Model(BaseModel):
    type: ModelType
    params: Union[
        CmKanModelParams,
    ]

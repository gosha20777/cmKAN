from pydantic import BaseModel
from enum import Enum
from typing import List, Union


class ModelType(str, Enum):
    cm_kan = "cm_kan"
    light_cm_kan = "light_cm_kan"
    cycle_cm_kan = "cycle_cm_kan"


class CmKanModelParams(BaseModel):
    in_dims: List[int] = [3]
    out_dims: List[int] = [3]
    grid_size: int = 5
    spline_order: int = 3
    residual_std: float = 0.1
    grid_range: List[float] = [0.0, 1.0]


class Model(BaseModel):
    type: ModelType = ModelType.cm_kan
    params: Union[CmKanModelParams,] = CmKanModelParams()

from ..config.model import ModelType
from ..config import Config
from typing import Union
from cm_kan.ml.models import (
    CmKAN,
    LightCmKAN,
    CycleCmKAN,
)


class ModelSelector:
    def select(config: Config) -> Union[CmKAN, LightCmKAN]:
        match config.model.type:
            case ModelType.cm_kan:
                return CmKAN(
                    in_dims=config.model.params.in_dims,
                    out_dims=config.model.params.out_dims,
                    grid_size=config.model.params.grid_size,
                    spline_order=config.model.params.spline_order,
                    residual_std=config.model.params.residual_std,
                    grid_range=config.model.params.grid_range
                )
            case ModelType.light_cm_kan:
                return LightCmKAN(
                    in_dims=config.model.params.in_dims,
                    out_dims=config.model.params.out_dims,
                    grid_size=config.model.params.grid_size,
                    spline_order=config.model.params.spline_order,
                    residual_std=config.model.params.residual_std,
                    grid_range=config.model.params.grid_range
                )
            case ModelType.cycle_cm_kan:
                return CycleCmKAN(
                    in_dims=config.model.params.in_dims,
                    out_dims=config.model.params.out_dims,
                    grid_size=config.model.params.grid_size,
                    spline_order=config.model.params.spline_order,
                    residual_std=config.model.params.residual_std,
                    grid_range=config.model.params.grid_range,
                )
            case _:
                raise ValueError(f'Unupported model type f{config.model.type}')

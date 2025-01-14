from ..config.model import ModelType
from ..config import Config
from typing import Union
from color_transfer.ml.models import (
    SimpleKanModel,
    ConvKanModel,
    MWRCAN,
    SepKAN,
    ConvSepKAN,
    SMS,
    SepKAN2D,
)


class ModelSelector:
    def select(config: Config) -> Union[SimpleKanModel, ConvKanModel, MWRCAN]:
        match config.model.type:
            case ModelType.simple_kan:
                return SimpleKanModel(
                    in_dims=config.model.params.in_dims,
                    out_dims=config.model.params.out_dims,
                    grid_size=config.model.params.grid_size,
                    spline_order=config.model.params.spline_order,
                    residual_std=config.model.params.residual_std,
                    grid_range=config.model.params.grid_range
                )
            case ModelType.conv_kan:
                return ConvKanModel(
                    in_channels=config.model.params.in_channels,
                    out_channels=config.model.params.out_channels,
                    grid_size=config.model.params.grid_size,
                    spline_order=config.model.params.spline_order,
                    residual_std=config.model.params.residual_std,
                    grid_range=config.model.params.grid_range
                )
            case ModelType.mw_isp:
                return MWRCAN()
            case ModelType.sep_kan:
                return SepKAN(
                    in_dims=config.model.params.in_dims,
                    out_dims=config.model.params.out_dims,
                    grid_size=config.model.params.grid_size,
                    spline_order=config.model.params.spline_order,
                    residual_std=config.model.params.residual_std,
                    grid_range=config.model.params.grid_range
                )
            case ModelType.conv_sep_kan:
                return ConvSepKAN(
                    in_dims=config.model.params.in_dims,
                    out_dims=config.model.params.out_dims,
                    kernel_sizes = config.model.params.kernel_sizes,
                    grid_size=config.model.params.grid_size,
                    spline_order=config.model.params.spline_order,
                    residual_std=config.model.params.residual_std,
                    grid_range=config.model.params.grid_range
                )
            case ModelType.sms:
                return SMS()
            case ModelType.sep_kan_2d:
                return SepKAN2D(
                    in_dims=config.model.params.in_dims,
                    out_dims=config.model.params.out_dims,
                    grid_size=config.model.params.grid_size,
                    spline_order=config.model.params.spline_order,
                    residual_std=config.model.params.residual_std,
                    grid_range=config.model.params.grid_range
                )
            case _:
                raise ValueError(f'Unupported model type f{config.model.type}')

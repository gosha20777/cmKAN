from ..config.pipeline import PipelineType
from ..config import Config
from typing import Union
from color_transfer.ml.pipelines import (
    DefaultPipeline,
    ImageSignalPipeline,
    MWImageSignalPipeline,
    SepKanPipeline,
    SMSPipeline,
    SepKan2DPipeline,
)
from color_transfer.ml.models import (
    SimpleKanModel,
    ConvKanModel,
    MWRCAN,
    SepKAN2D
)


class PipelineSelector:
    def select(config: Config, model: Union[SimpleKanModel, ConvKanModel, MWRCAN, SepKAN2D]) -> Union[DefaultPipeline, ImageSignalPipeline, MWImageSignalPipeline]:
        match config.pipeline.type:
            case PipelineType.default:
                return DefaultPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay
                )
            case PipelineType.isp:
                return ImageSignalPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay
                )
            case PipelineType.mw_isp:
                return MWImageSignalPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay
                )
            case PipelineType.sep_kan:
                return SepKanPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay
                )
            case PipelineType.sms:
                return SMSPipeline(model=model)
            case PipelineType.sep_kan_2d:
                return SepKan2DPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay
                )
            case _:
                raise ValueError(f'Unupported pipeline type f{config.pipeline.type}')

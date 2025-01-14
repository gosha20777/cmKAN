from ..config.pipeline import PipelineType
from ..config import Config
from typing import Union
from cm_kan.ml.pipelines import (
    SupervisedPipeline,
    UnsupervisedPipeline,
    PairBasedPipeline,
)
from cm_kan.ml.models import (
    CmKAN,
)


class PipelineSelector:
    def select(config: Config, model: Union[CmKAN]) -> Union[SupervisedPipeline, UnsupervisedPipeline, PairBasedPipeline]:
        match config.pipeline.type:
            case PipelineType.supervised:
                return SupervisedPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay
                )
            case PipelineType.unsupervised:
                return UnsupervisedPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay
                )
            case PipelineType.pair_based:
                return PairBasedPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay
                )
            case _:
                raise ValueError(f'Unupported pipeline type f{config.pipeline.type}')

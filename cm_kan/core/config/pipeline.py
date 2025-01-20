from enum import Enum
from pydantic import BaseModel
from typing import Union


class PipelineType(str, Enum):
    supervised   = 'supervised'
    unsupervised = 'unsupervised'
    pair_based   = 'pair_based'
    

class PipelineParams(BaseModel):
    lr: float = 1e-3
    batch_size: int = 32
    val_batch_size: int = 1
    test_batch_size: int = 1
    epochs: int = 500
    save_freq: int = 10
    visualize_freq: int = 10


class DefaultPipelineParams(PipelineParams):
    optimizer: str = 'adam'
    weight_decay: float = 0.0

class PairBasedPipelineParams(DefaultPipelineParams):
    finetune_iters: int = 10


class Pipeline(BaseModel):
    type: PipelineType = PipelineType.supervised
    params: Union[
        PairBasedPipelineParams,
        DefaultPipelineParams,
    ]

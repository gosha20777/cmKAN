from ..config.data import DataType
from ..config import Config
from typing import Union
from cm_kan.ml.datasets import (
    Valga2kImgDataModule,
    Valga2kImgUnpairedDataModule,
    FiveKImgDataModule,
)


class DataSelector:
    def select(config: Config) -> Union[Valga2kImgDataModule, FiveKImgDataModule]:
        match config.data.type:
            case DataType.volga2k:
                return Valga2kImgDataModule(
                    train_a=config.data.train.source,
                    train_b=config.data.train.target,
                    val_a=config.data.val.source,
                    val_b=config.data.val.target,
                    test_a=config.data.test.source,
                    test_b=config.data.test.target,
                    batch_size=config.pipeline.params.batch_size,
                    val_batch_size=config.pipeline.params.val_batch_size,
                    test_batch_size=config.pipeline.params.test_batch_size,
                )
            case DataType.volga2k_unpaired:
                return Valga2kImgUnpairedDataModule(
                    train_a=config.data.train.source,
                    train_b=config.data.train.target,
                    val_a=config.data.val.source,
                    val_b=config.data.val.target,
                    test_a=config.data.test.source,
                    test_b=config.data.test.target,
                    batch_size=config.pipeline.params.batch_size,
                    val_batch_size=config.pipeline.params.val_batch_size,
                    test_batch_size=config.pipeline.params.test_batch_size,
                )
            case DataType.five_k:
                return FiveKImgDataModule(
                    train_a=config.data.train.source,
                    train_b=config.data.train.target,
                    val_a=config.data.val.source,
                    val_b=config.data.val.target,
                    test_a=config.data.test.source,
                    test_b=config.data.test.target,
                    batch_size=config.pipeline.params.batch_size,
                    val_batch_size=config.pipeline.params.val_batch_size,
                    test_batch_size=config.pipeline.params.test_batch_size,
                )
            case _:
                raise ValueError(f"Unupported data type f{config.data.type}")

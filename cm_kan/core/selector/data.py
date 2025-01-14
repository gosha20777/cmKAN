from ..config.data import DataType
from ..config import Config
from typing import Union
from color_transfer.ml.datasets import (
    HuaweiDataModule,
    HuaweiImgDataModule,
    HuaweiImgFeatDataModule,
    HuaweiSMSDataModule,
    FiveKImgDataModule,
)


class DataSelector:
    def select(config: Config) -> Union[HuaweiDataModule, HuaweiImgDataModule]:
        match config.data.type:
            case DataType.huawei:
                return HuaweiDataModule(
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
            case DataType.huawei_img:
                return HuaweiImgDataModule(
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
            case DataType.huawei_img_feat:
                return HuaweiImgFeatDataModule(
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
            case DataType.huawei_sms:
                return HuaweiSMSDataModule(
                    path_a=config.data.test.source,
                    path_b=config.data.test.target,
                    batch_size=config.pipeline.params.batch_size,
                )
            case DataType.five_k_img:
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
                raise ValueError(f'Unupported data type f{config.data.type}')
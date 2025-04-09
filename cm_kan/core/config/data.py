from pydantic import BaseModel
from enum import Enum


class DataType(str, Enum):
    huawei = 'huawei'
    huawei_unpaired = 'huawei_unpaired'
    five_k = 'five_k'  


class DataPathes(BaseModel):
    source: str
    target: str


class Data(BaseModel):
    type: DataType = DataType.huawei
    train: DataPathes
    val: DataPathes
    test: DataPathes
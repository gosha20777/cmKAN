from pydantic import BaseModel
from enum import Enum


class DataType(str, Enum):
    huawei = 'huawei' 
    huawei_img = 'huawei_img' 
    huawei_img_feat = 'huawei_img_feat' 
    huawei_sms = 'huawei_sms'
    five_k_img = 'five_k_img'  


class DataPathes(BaseModel):
    source: str
    target: str


class Data(BaseModel):
    type: DataType = DataType.huawei
    train: DataPathes
    val: DataPathes
    test: DataPathes
from typing import Union
from pydantic import BaseModel

from rich.syntax import Syntax
from rich import print

from .data import Data
from .model import Model
from .pipeline import Pipeline

import yaml


class Config(BaseModel):
    experiment: str = 'huawei_supervised'
    save_dir: str = 'experiments'
    resume: bool = False
    model: Model
    data: Data
    pipeline: Pipeline = Pipeline()
    accelerator: Union[str,int] = 'gpu'

    def print(self) -> None:
        str = yaml.dump(self.model_dump())
        syntax = Syntax(str, "yaml")
        print(syntax)

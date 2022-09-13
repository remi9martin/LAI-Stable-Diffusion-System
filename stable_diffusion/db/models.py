import inspect
import sys
from typing import Optional

from pydantic import BaseModel
from sqlmodel import Field, SQLModel

from stable_diffusion.utilities.enum import Stage


class GeneralModel(BaseModel):
    cls_name: str
    cls_module: str
    data: str

    def convert_to_model(self):
        return self.data_cls.parse_raw(self.data)

    @property
    def data_cls(self) -> BaseModel:
        return getattr(sys.modules[self.cls_module], self.cls_name)

    @classmethod
    def from_obj(cls, obj):
        return cls(
            **{
                "cls_path": inspect.getfile(obj.__class__),
                "cls_name": obj.__class__.__name__,
                "cls_module": obj.__class__.__module__,
                "data": obj.json(),
            }
        )

    @classmethod
    def from_cls(cls, obj_cls):
        return cls(
            **{
                "cls_path": inspect.getfile(obj_cls),
                "cls_name": obj_cls.__name__,
                "cls_module": obj_cls.__module__,
                "data": "",
            }
        )


class PromptConfig(SQLModel, table=True):

    id: Optional[int] = Field(default=None, primary_key=True)
    prompt: str
    style: str
    num_images: int
    stage: str = Stage.NOT_STARTED

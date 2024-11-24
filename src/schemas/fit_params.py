from pydantic import BaseModel, Field
from typing import Optional, List
from typing_extensions import Annotated
import pandas as pd

from src.neural_net.callbacks.callback import Callback


class FitParams(BaseModel):
    X: pd.DataFrame
    epochs: Annotated[int, Field(strict=True, ge=1)]
    val_split: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        None, description="Fraction of data for validation (0 < val_split < 1)"
    )
    val_data: pd.DataFrame = Field(
        None, description="Validation data must be a pandas DataFrame"
    )
    callbacks: Annotated[List[Callback], Field(default_factory=list)] = Field(
        description="List of callback objects"
    )
    batch_size: Annotated[int, Field(strict=True, ge=1)] = 32
    verbose: bool = False

    class Config:
        arbitrary_types_allowed = True


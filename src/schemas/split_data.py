from pydantic import BaseModel
from numpy.typing import NDArray
import numpy as np


class SplitData(BaseModel):
    X_train: NDArray[np.float64]
    y_train: NDArray[np.float64]
    X_val: NDArray[np.float64]
    y_val: NDArray[np.float64]

    class Config:
        arbitrary_types_allowed = True
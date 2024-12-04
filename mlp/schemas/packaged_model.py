from pydantic import BaseModel

from sklearn.preprocessing import StandardScaler

from mlp.neural_net.core.sequential import Sequential
from mlp.schemas.csv_labels import CSVColNames
from mlp.schemas.sequential_config import SequentialModelConfig


class PackagedModel(BaseModel):
    model: Sequential
    scaler: StandardScaler
    df_col_names: CSVColNames
    binary_target_map: dict
    config: SequentialModelConfig

    class Config:
        arbitrary_types_allowed = True

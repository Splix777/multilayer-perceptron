import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler


class ProcessedData(BaseModel):
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    scaler: StandardScaler
    binary_target_map: dict

    class Config:
        arbitrary_types_allowed = True
from pandas import DataFrame
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler


class ProcessedData(BaseModel):
    train_df: DataFrame
    val_df: DataFrame
    scaler: StandardScaler
    target_encoding_map: dict

    class Config:
        arbitrary_types_allowed = True

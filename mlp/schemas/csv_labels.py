from typing import List
from pydantic import BaseModel


class CSVColNames(BaseModel):
    id: str
    target: str
    features: List[str]

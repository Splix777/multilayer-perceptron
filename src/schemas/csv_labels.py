from typing import List
from pydantic import BaseModel

class CSVLabels(BaseModel):
    id: str
    target: str
    features: List[str]

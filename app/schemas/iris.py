
from pydantic import BaseModel, Field
from typing import Dict, List

class FlowerDims(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=20, description="Sepal length (cm)")
    sepal_width:  float = Field(..., gt=0, lt=20, description="Sepal width (cm)")
    petal_length: float = Field(..., gt=0, lt=20, description="Petal length (cm)")
    petal_width:  float = Field(..., gt=0, lt=20, description="Petal width (cm)")

class IrisOutput(BaseModel):
    predicted_class_index: int
    predicted_class_label: str
    probabilities: Dict[str, float]
    feature_order: List[str]


from pydantic import BaseModel, Field
from typing import Dict, List

class WineInput(BaseModel):
    volatile_acidity: float = Field(..., gt=0, lt=2, description="volatile acidity (g/L)")
    total_sulfur_dioxide:  float = Field(..., gt=0, lt=200, description="total sulfur dioxide (mg/L)")
    sulphates: float = Field(..., gt=0, lt=1.5, description="sulphates (mg/cL)") # à convertir en mg/L au frontend (donc à diviser par 100 avant d'envoyer du frontend à l'API)
    alcohol:  float = Field(..., gt=6, lt=16, description="alcohol (vol%)")

class WineOutput(BaseModel):
    quality: float
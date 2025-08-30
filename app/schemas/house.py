from pydantic import BaseModel, Field

class HouseInput(BaseModel):
    GrLivArea: float = Field(..., gt=100, lt=10000, description="Above ground living area (ft²)")
    OverallQual: int  = Field(..., ge=1, le=10, description="Overall quality (1-10)")
    YearBuilt:  int   = Field(..., ge=1870, le=2025, description="Year built")
    GarageCars: int   = Field(..., ge=0, le=5, description="Garage capacity (cars)")
    FullBath:   int   = Field(..., ge=0, le=5, description="Full bathrooms")
    LotArea:    float = Field(..., gt=500, lt=1_000_000, description="Lot area (ft²)")

class HouseOutput(BaseModel):
    price_usd: float
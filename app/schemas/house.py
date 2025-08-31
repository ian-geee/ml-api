from pydantic import BaseModel, Field

class HouseInput(BaseModel):
    habitable_surface: float = Field(..., gt=50, lt=400, description="Above ground living area (m²)")
    bedrooms_count: int = Field(..., ge=1, le=6, description="Number of bedrooms")
    post_code: str = Field(..., description="Postal code")
    num_facade: int = Field(..., ge=1, le=4, description="Number of facades")

class HouseOutput(BaseModel):
    price_euro: float
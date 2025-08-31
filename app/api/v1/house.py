from fastapi import APIRouter, Depends, HTTPException, Request
from app.schemas.house import HouseInput, HouseOutput
from app.services.house_service import HouseService

router = APIRouter(prefix="/api/v1/house", tags=["house"])

def get_house_service(request: Request) -> HouseService:
    svc = getattr(request.app.state, "house_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="House service not ready")
    return svc

@router.get("/health")
def health(svc: HouseService = Depends(get_house_service)):
    return {"status":"ok", "features": svc.feature_order}

@router.post("/predict", response_model=HouseOutput)
def predict(payload: HouseInput, svc: HouseService = Depends(get_house_service)):
    price = svc.predict(payload)
    return {"price_euro": price}
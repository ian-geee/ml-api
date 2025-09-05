
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from app.schemas.wine import WineInput, WineOutput
from app.services.wine_service import WineService
from app.core.config import settings

router = APIRouter(prefix="/api/v1/wine", tags=["wine"])

def get_wine_service(request: Request) -> WineService:
    svc = getattr(request.app.state, "wine_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Wine service not ready")
    return svc

def check_api_key(x_api_key: str | None = Header(default=None, convert_underscores=False)):
    required = settings.ml_api_key
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid API key")

@router.get("/health")
def health(svc: WineService = Depends(get_wine_service)):
    return {"status":"ok", "features": svc.feature_order}

@router.post("/predict", response_model=WineOutput)
def predict(payload: WineInput, svc: WineService = Depends(get_wine_service)):
    quality = svc.predict(payload)
    return {"quality": quality} # mandatory to return a dict, return svc.predict(payload) return une erreur
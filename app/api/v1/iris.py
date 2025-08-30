
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from app.schemas.iris import FlowerDims, IrisOutput
from app.services.iris_service import IrisService
from app.core.config import settings

router = APIRouter(prefix="/api/v1/iris", tags=["iris"])

def get_iris_service(request: Request) -> IrisService:
    svc = getattr(request.app.state, "iris_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Iris service not ready")
    return svc

def check_api_key(x_api_key: str | None = Header(default=None, convert_underscores=False)):
    required = settings.ml_api_key
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid API key")

@router.get("/health")
def health(svc: IrisService = Depends(get_iris_service)):
    return {
        "status": "ok",
        "model": svc.meta.get("model_type"),
        "model_version": svc.meta.get("model_version"),
        "classes": svc.class_names,
    }

@router.post("/predict", response_model=IrisOutput)
def predict(payload: FlowerDims, svc: IrisService = Depends(get_iris_service), _: None = Depends(check_api_key)):
    return svc.predict(payload)

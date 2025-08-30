
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings

from app.services.iris_service import IrisService
from app.api.v1.iris import router as iris_router

from app.services.house_service import HouseService
from app.api.v1.house import router as house_router

class BodySizeLimiter(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int = 8 * 1024):
        super().__init__(app); self.max = max_bytes
    async def dispatch(self, request, call_next):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > self.max:
            return JSONResponse({"detail":"Payload too large"}, status_code=413)
        body = await request.body()
        if body and len(body) > self.max:
            return JSONResponse({"detail":"Payload too large"}, status_code=413)
        return await call_next(request)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.iris_service = IrisService(settings.model_dir)   # FileNotFoundError -> crash
    app.state.house_service = HouseService(settings.model_dir) # idem
    yield

app = FastAPI(title="Portfolio API", version="v1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET","POST","OPTIONS","HEAD"],
    allow_headers=["content-type","x-api-key"],
)
app.add_middleware(BodySizeLimiter)

@app.get("/")
def root():
    return {"ok": True, "message": "Use GET /health or POST /api/v1/iris/predict"}

@app.get("/health")
def health():
    svc = getattr(app.state, "iris_service", None)
    return {
        "status": "ok" if svc else "booting",
        "iris_loaded": bool(svc),
        "version": app.version,
    }

app.include_router(iris_router)
app.include_router(house_router)

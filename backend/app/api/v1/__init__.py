from fastapi import APIRouter
from app.api.v1.recordings import router as recordings_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(recordings_router)

__all__ = ["api_v1_router"]

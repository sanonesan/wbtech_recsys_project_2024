"""
Assemble routers
"""

from fastapi import APIRouter

from src.api.routers.data import data_router
from src.api.routers.trainning import trainning_router
from src.api.routers.inference import inference_router
from src.api.routers.pipelines import pipelines_router

# Create generalizing router
router = APIRouter(prefix="/wbtech_proj")

# Include routers
router.include_router(data_router)
router.include_router(trainning_router)
router.include_router(inference_router)
router.include_router(pipelines_router)


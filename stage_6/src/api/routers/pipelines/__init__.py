"""
Assemble pipelines routers
"""

from fastapi import APIRouter


from src.api.routers.pipelines.preprocess_raw_data import (
    preprocessing_raw_data_pipeline_router,
)
from src.api.routers.pipelines.train import trainning_pipeline_router
from src.api.routers.pipelines.inference import inference_pipeline_router

# Create generalizing router
pipelines_router = APIRouter(prefix="/pipelines")

# Include routers
pipelines_router.include_router(preprocessing_raw_data_pipeline_router)
pipelines_router.include_router(trainning_pipeline_router)
pipelines_router.include_router(inference_pipeline_router)

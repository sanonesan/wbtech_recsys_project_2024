"""
Assemble pipelines routers
"""

from fastapi import APIRouter


from src.api.routers.pipelines.preprocess_raw_data import (
    preprocessing_raw_data_pipeline_router,
)
from src.api.routers.pipelines.train import train_ranker_pipeline_router
from src.api.routers.pipelines.inference import inference_ranker_pipeline_router

# Create generalizing router
pipelines_router = APIRouter(prefix="/pipelines")

# Include routers
pipelines_router.include_router(preprocessing_raw_data_pipeline_router)
pipelines_router.include_router(train_ranker_pipeline_router)
pipelines_router.include_router(inference_ranker_pipeline_router)

"""
Assemble trainning routers
"""

from fastapi import APIRouter


from src.api.routers.trainning.first_stage import (
    trainning_first_stage_for_ranker_trainning_router,
    trainning_first_stage_for_ranker_inference_router,
)
from src.api.routers.trainning.second_stage import trainning_second_stage_router
from src.api.routers.trainning.cold_recs import trainning_cold_recs_models_router


# Create generalizing router
trainning_router = APIRouter(prefix="/train_models")

# Include routers
trainning_router.include_router(trainning_first_stage_for_ranker_trainning_router)
trainning_router.include_router(trainning_first_stage_for_ranker_inference_router)

trainning_router.include_router(trainning_second_stage_router)

trainning_router.include_router(trainning_cold_recs_models_router)

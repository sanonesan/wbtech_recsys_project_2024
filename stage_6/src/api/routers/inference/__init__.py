"""
Assemble inference routers
"""

from fastapi import APIRouter


from src.api.routers.inference.first_stage import (
    inference_first_stage_for_ranker_trainning_router,
    inference_first_stage_for_ranker_inference_router,
)
from src.api.routers.inference.second_stage import (
    inference_second_stage_router
)


# Create generalizing router
inference_router = APIRouter(prefix="/inference_models")

# Include routers
inference_router.include_router(inference_first_stage_for_ranker_trainning_router)
inference_router.include_router(inference_first_stage_for_ranker_inference_router)

inference_router.include_router(inference_second_stage_router)


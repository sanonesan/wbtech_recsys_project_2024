"""
Assemble preprocessing routers
"""

from fastapi import APIRouter


from src.api.routers.data.preprocessing.raw_data import preprocessing_raw_data_router
from src.api.routers.data.preprocessing.ranker import preprocessing_ranker_data_router



# Create generalizing router
preprocessing_router = APIRouter(prefix="/preprocessing")

# Include routers
preprocessing_router.include_router(preprocessing_raw_data_router)
preprocessing_router.include_router(preprocessing_ranker_data_router)

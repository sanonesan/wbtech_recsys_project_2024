"""
Assemble data routers
"""

from fastapi import APIRouter


from src.api.routers.data.preprocessing import preprocessing_router


# Create generalizing router
data_router = APIRouter(prefix="/data")

# Include routers
data_router.include_router(preprocessing_router)

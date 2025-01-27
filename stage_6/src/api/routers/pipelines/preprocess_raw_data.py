# pylint: disable=R0801
"""
Router for preprocessing raw data
"""

from fastapi import APIRouter, status, Response


from src.configurations.settings import SETTINGS

from src.api.routers.data.preprocessing.raw_data import preprocess_all_data

preprocessing_raw_data_pipeline_router = APIRouter(tags=["Pipelines"], prefix="/raw_data")


@preprocessing_raw_data_pipeline_router.post(
    path="/preprocess/",
    status_code=status.HTTP_200_OK,
)
async def preprocess_raw_data_pipeline(
    interactions_path: str = SETTINGS.interactions_path,
    text_data_path: str = SETTINGS.text_data_path,
    imgs_data_path: str = SETTINGS.imgs_data_path,
    data_path: str = SETTINGS.data_path,
    batch_size: int = SETTINGS.imgs_batch_size,
) -> Response:
    """ """

    try:
        response = await preprocess_all_data(
            interactions_path=interactions_path,
            text_data_path=text_data_path,
            imgs_data_path=imgs_data_path,
            data_path=data_path,
            batch_size=batch_size,
        )
        return response

    except Exception as e:
        return Response(
            content=f"Error while preprocessing raw data: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )

# pylint: disable=R0801
"""
Router for preprocessing raw data
"""

from fastapi import APIRouter, status, Response


from src.configurations.settings import SETTINGS

from src.ml.data.preprocessing.raw.interactions import preprocess_interactions_pipeline
from src.ml.data.preprocessing.raw.items_text import preprocess_items_text_data_pipline
from src.ml.data.preprocessing.raw.items_images import (
    preprocess_items_images_data_pipline,
)


preprocessing_raw_data_router = APIRouter(
    tags=["PreprocessRawData"], prefix="/raw_data"
)


@preprocessing_raw_data_router.post(
    path="/interactions/",
    status_code=status.HTTP_200_OK,
)
async def preprocess_interactions(
    interactions_path: str = SETTINGS.interactions_path,
    data_path: str = SETTINGS.data_path,
) -> Response:
    """
    Preprocesses raw interaction data.

    Args:
        interactions_path: Path to the raw interactions data.
        data_path: Path to store the preprocessed data.

    Returns:
        200 OK if successful, 409 CONFLICT if an error occurs during preprocessing.
    """

    try:
        preprocess_interactions_pipeline(
            interactions_path=interactions_path,
            data_path=data_path,
        )
        return Response(
            content="Interactions data preprocessed!", status_code=status.HTTP_200_OK
        )

    except Exception as e:
        return Response(
            content=f"Error while preprocessing interactions data: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


@preprocessing_raw_data_router.post(
    path="/text_data/",
    status_code=status.HTTP_200_OK,
)
async def preprocess_items_text_data(
    text_data_path: str = SETTINGS.text_data_path,
    data_path: str = SETTINGS.data_path,
) -> Response:
    """
    Preprocesses raw text data associated with items.

    Args:
        text_data_path: Path to the raw text data file.
        data_path: Path to store the preprocessed text data.

    Returns:
        200 OK if successful, 409 CONFLICT if an error occurs during preprocessing.
    """

    try:
        preprocess_items_text_data_pipline(
            text_data_path=text_data_path,
            data_path=data_path,
        )
        return Response(
            content="Items text data preprocessed!", status_code=status.HTTP_200_OK
        )

    except Exception as e:
        return Response(
            content=f"Error while preprocessing items text data: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


@preprocessing_raw_data_router.post(
    path="/images_data/",
    status_code=status.HTTP_200_OK,
)
async def preprocess_items_images_data(
    imgs_data_path: str = SETTINGS.imgs_data_path,
    data_path: str = SETTINGS.data_path,
    batch_size: int = SETTINGS.imgs_batch_size,
) -> Response:
    """
    Preprocesses raw image data associated with items.

    Args:
        imgs_data_path: Path to the raw images data directory.
        data_path: Path to store the preprocessed image data.
        batch_size: Number of images to process in each batch.

    Returns:
        200 OK if successful, 409 CONFLICT if an error occurs during preprocessing.
    """

    try:
        preprocess_items_images_data_pipline(
            imgs_data_path=imgs_data_path,
            data_path=data_path,
            batch_size=batch_size,
        )
        return Response(
            content="Items images data preprocessed!", status_code=status.HTTP_200_OK
        )

    except Exception as e:
        return Response(
            content=f"Error while preprocessing items images data: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


@preprocessing_raw_data_router.post(
    path="/full/",
    status_code=status.HTTP_200_OK,
)
async def preprocess_all_data(
    interactions_path: str = SETTINGS.interactions_path,
    text_data_path: str = SETTINGS.text_data_path,
    imgs_data_path: str = SETTINGS.imgs_data_path,
    data_path: str = SETTINGS.data_path,
    batch_size: int = SETTINGS.imgs_batch_size,
) -> Response:
    """
    Preprocesses all raw data including interactions, text, and images.

    Args:
        interactions_path: Path to the raw interactions data.
        text_data_path: Path to the raw text data file.
        imgs_data_path: Path to the raw images data directory.
        data_path: Path to store the preprocessed data.
        batch_size: Number of images to process in each batch.

    Returns:
        200 OK if successful, 409 CONFLICT if an error occurs during preprocessing.
    """

    try:
        await preprocess_interactions(
            interactions_path=interactions_path, data_path=data_path
        )
        await preprocess_items_text_data(
            text_data_path=text_data_path, data_path=data_path
        )
        await preprocess_items_images_data(
            imgs_data_path=imgs_data_path,
            data_path=data_path,
            batch_size=batch_size,
        )
        return Response(
            content="All raw data preprocessed!", status_code=status.HTTP_200_OK
        )

    except Exception as e:
        return Response(
            content=f"Error while preprocessing raw data: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )

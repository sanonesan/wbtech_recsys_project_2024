# pylint: disable=R0801
"""
Router for preprocessing data for inference cold recs
"""

from fastapi import APIRouter, status, Response

from src.configurations.settings import SETTINGS

from src.ml.models.cold_recs.inferencer import Inferencer

inference_cold_recs_models_router = APIRouter(
    tags=["InferenceColdRecsModels"], prefix="/cold_recs_models"
)


@inference_cold_recs_models_router.post(
    path="/popular_model/",
    status_code=status.HTTP_201_CREATED,
)
async def inference_popular_model(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
) -> Response:
    """
    Endpoint to train the popular model.

    Args:
        data_path: Path to the directory containing input data (default from SETTINGS).
        models_path: Path to the directory where trained models are saved (default from SETTINGS).
        candidates_data_path: Path to directory to save generated recommendations 
            (default from SETTINGS).

    Returns:
        A Response object containing a message indicating success or failure of the training
        process and an appropriate HTTP status code.
    """

    try:

        Inferencer.inference_popular_model(
            data_path=data_path,
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

        return Response(
            content="Popular model trained!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while trainning Popular model: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


@inference_cold_recs_models_router.post(
    path="/popular_bandit_model/",
    status_code=status.HTTP_201_CREATED,
)
async def inference_popular_bandit_model(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
) -> Response:
    """
    Endpoint to iference the popular bandit model.

    Args:
        data_path: Path to the directory containing input data (default from SETTINGS).
        models_path: Path to the directory where trained models are saved (default from SETTINGS).
        candidates_data_path: Path to directory to save generated recommendations 
            (default from SETTINGS).

    Returns:
        A Response object containing a message indicating success or failure of the training
        process and an appropriate HTTP status code.
    """

    try:

        Inferencer.inference_popular_bandit_model(
            data_path=data_path,
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

        return Response(
            content="Popular model trained!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while trainning Popular model: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )

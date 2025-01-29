# pylint: disable=R0801
"""
Router for preprocessing raw data
"""

from fastapi import APIRouter, status, Response


from src.configurations.settings import SETTINGS

from src.api.routers.trainning.first_stage import train_all_models_for_ranker_trainning
from src.api.routers.inference.first_stage import (
    inference_all_models_for_ranker_trainning,
)
from src.api.routers.data.preprocessing.ranker import (
    preprocess_data_for_ranker_trainning,
)
from src.api.routers.trainning.second_stage import train_catboost_model

from src.api.routers.trainning.cold_recs import train_popular_model, fit_mab_model
from src.api.routers.inference.cold_recs import inference_popular_model

trainning_pipeline_router = APIRouter(tags=["Pipelines"], prefix="/train")


@trainning_pipeline_router.post(
    path="/ranker/",
    status_code=status.HTTP_200_OK,
)
async def trainning_ranker_pipeline(
    models_path: str = SETTINGS.models_path,
    data_path: str = SETTINGS.data_path,
    candidates_data_path: int = SETTINGS.candidates_data_path,
) -> Response:
    """
    Executes the complete ranker training pipeline.

    Args:
        models_path: Path to store the trained models.
        data_path: Path to the training data.
        candidates_data_path: Path to the candidate data.

    Returns:
        200 OK if successful, 409 CONFLICT if an error occurs during
        the pipeline execution.
    """

    try:

        await train_all_models_for_ranker_trainning(
            data_path=data_path,
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

        await inference_all_models_for_ranker_trainning(
            data_path=data_path,
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

        await preprocess_data_for_ranker_trainning(
            data_path=data_path,
            candidates_data_path=candidates_data_path,
        )

        await train_catboost_model(
            data_path=data_path,
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

        return Response(
            content="Trainning pipeline completed!",
            status_code=status.HTTP_409_CONFLICT,
        )

    except Exception as e:
        return Response(
            content=f"Error while trainning pipeline: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


@trainning_pipeline_router.post(
    path="/cold_recs/",
    status_code=status.HTTP_200_OK,
)
async def trainning_cold_recs_pipeline(
    models_path: str = SETTINGS.models_path,
    data_path: str = SETTINGS.data_path,
    candidates_data_path: int = SETTINGS.candidates_data_path,
) -> Response:
    """
    Executes the complete cold recs training pipeline.

    Args:
        models_path: Path to store the trained models.
        data_path: Path to the training data.
        candidates_data_path: Path to the candidate data.

    Returns:
        200 OK if successful, 409 CONFLICT if an error occurs during
        the pipeline execution.
    """

    try:

        await train_popular_model(
            data_path=data_path,
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

        await inference_popular_model(
            data_path=data_path,
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

        await fit_mab_model(
            data_path=data_path,
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

        return Response(
            content="Trainning pipeline completed!",
            status_code=status.HTTP_409_CONFLICT,
        )

    except Exception as e:
        return Response(
            content=f"Error while trainning pipeline: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )

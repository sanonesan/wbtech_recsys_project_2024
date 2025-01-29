# pylint: disable=R0801
"""
Router for preprocessing raw data
"""

from fastapi import APIRouter, status, Response


from src.configurations.settings import SETTINGS

from src.api.routers.trainning.first_stage import train_all_models_for_ranker_inference
from src.api.routers.inference.first_stage import (
    inference_all_models_for_ranker_inference,
)
from src.api.routers.data.preprocessing.ranker import (
    preprocess_data_for_ranker_inference,
)
from src.api.routers.inference.second_stage import inference_catboost_model
from src.api.routers.inference.cold_recs import inference_popular_bandit_model

inference_pipeline_router = APIRouter(tags=["Pipelines"], prefix="/inference")


@inference_pipeline_router.post(
    path="/ranker/",
    status_code=status.HTTP_200_OK,
)
async def inference_ranker_pipeline(
    models_path: str = SETTINGS.models_path,
    data_path: str = SETTINGS.data_path,
    candidates_data_path: int = SETTINGS.candidates_data_path,
) -> Response:
    """
    Executes the complete ranker inference pipeline.

    Args:
        models_path: Path to the directory containing the trained models.
        data_path: Path to the data directory.
        candidates_data_path: Path to the candidate data.

    Returns:
        200 OK if successful, 409 CONFLICT if an error occurs
        during pipeline execution.
    """

    try:

        await train_all_models_for_ranker_inference(
            data_path=data_path,
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

        await inference_all_models_for_ranker_inference(
            data_path=data_path,
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

        await preprocess_data_for_ranker_inference(
            data_path=data_path,
            candidates_data_path=candidates_data_path,
        )

        await inference_catboost_model(
            data_path=data_path,
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

        return Response(
            content="Inference pipeline completed!",
            status_code=status.HTTP_409_CONFLICT,
        )

    except Exception as e:
        return Response(
            content=f"Error while inference pipeline: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


@inference_pipeline_router.post(
    path="/cold_recs/",
    status_code=status.HTTP_200_OK,
)
async def inference_pop_mab_pipeline(
    models_path: str = SETTINGS.models_path,
    data_path: str = SETTINGS.data_path,
    candidates_data_path: int = SETTINGS.candidates_data_path,
) -> Response:
    """
    Executes the complete popular bandit inference pipeline.

    Args:
        models_path: Path to the directory containing the trained models.
        data_path: Path to the data directory.
        candidates_data_path: Path to the candidate data.

    Returns:
        200 OK if successful, 409 CONFLICT if an error occurs
        during pipeline execution.
    """

    try:

        await inference_popular_bandit_model(
            data_path=data_path,
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

    except Exception as e:
        return Response(
            content=f"Error while inference pipeline: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )

# pylint: disable=R0801
"""
Router for preprocessing data for trainnig second stage
"""

from fastapi import APIRouter, status, Response

from src.configurations.settings import SETTINGS

from src.ml.models.second_stage.inferencer import Inferencer

inference_second_stage_router = APIRouter(
    tags=["InferenceRanker"], prefix="/ranker_model"
)


@inference_second_stage_router.post(
    path="/catboost_model/",
    status_code=status.HTTP_201_CREATED,
)
async def inference_catboost_model(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
    model_name: str = "cb_ranker",
    n_splits: int = 5,
) -> Response:
    """ """

    try:

        Inferencer.inference_catboost_model(
            data_path=data_path,
            ranker_data="ranker_data_bNr.parquet",
            models_path=models_path,
            candidates_data_path=candidates_data_path,
            model_name=model_name,
            n_splits=n_splits,
        )

        return Response(
            content="CatBoost Ranker model inferenced!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while inferencing CatBoost Ranker model: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )

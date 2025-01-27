# pylint: disable=R0801
"""
Router for preprocessing data for trainnig second stage
"""

from fastapi import APIRouter, status, Response

from src.configurations.settings import SETTINGS

from src.ml.models.second_stage.trainner import Trainner

trainning_second_stage_router = APIRouter(
    tags=["TrainningRanker"], prefix="/ranker_model"
)


@trainning_second_stage_router.post(
    path="/catboost_model/",
    status_code=status.HTTP_201_CREATED,
)
async def train_catboost_model(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
    model_name: str = "cb_ranker",
) -> Response:
    """ """

    try:

        Trainner.train_catboost_model(
            data_path=data_path,
            models_path=models_path,
            candidates_data_path=candidates_data_path,
            model_name=model_name,
        )

        return Response(
            content="CatBoost Ranker model trained!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while trainning CatBoost Ranker model: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )

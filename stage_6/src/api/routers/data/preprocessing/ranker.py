# pylint: disable=R0801
"""
Router for preprocessing data for trainnig second stage
"""

from fastapi import APIRouter, status, Response

# from src.configurations.settings import SETTINGS

from src.configurations.settings import SETTINGS

from src.ml.data.preprocessing.ranker import (
    preprocess_data_for_ranker_trainning,
    preprocess_data_for_ranker_inference,
)

preprocessing_ranker_data_router = APIRouter(
    tags=["PreprocessDataForRanker"], prefix="/ranker_data"
)


@preprocessing_ranker_data_router.post(
    path="/for_trainning/",
    status_code=status.HTTP_201_CREATED,
)
async def preprocess_data_for_trainning(
    data_path=SETTINGS.data_path,
    candidates_data_path=SETTINGS.candidates_data_path,
    do_ranker_test=SETTINGS.do_ranker_test,
) -> Response:
    """ """

    try:
        preprocess_data_for_ranker_trainning(
            data_path=data_path,
            candidates_data_path=candidates_data_path,
            do_ranker_test=do_ranker_test,
        )
        return Response(
            content="Data for trainnning ranker preprocessed!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while preprocessing Data for trainnning ranker: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


@preprocessing_ranker_data_router.post(
    path="/for_inference/",
    status_code=status.HTTP_201_CREATED,
)
async def preprocess_data_for_inference(
    data_path=SETTINGS.data_path,
    candidates_data_path=SETTINGS.candidates_data_path,
) -> Response:
    """ """

    try:
        preprocess_data_for_ranker_inference(
            data_path=data_path,
            candidates_data_path=candidates_data_path,
        )
        return Response(
            content="Data for inferencing ranker preprocessed!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while preprocessing data for inferencing ranker: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )

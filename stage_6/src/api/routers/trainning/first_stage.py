# pylint: disable=R0801
"""
Router for preprocessing data for trainnig second stage
"""

from fastapi import APIRouter, status, Response

import polars as pl

from src.configurations.settings import SETTINGS

from src.ml.models.model_utils import create_rectools_dataset
from src.ml.models.first_stage.trainner import Trainner


trainning_first_stage_for_ranker_trainning_router = APIRouter(
    tags=["TrainningFirstStageForRankerTrainning"], prefix="/first_stage_models_for_ranker_trainning"
)

trainning_first_stage_for_ranker_inference_router = APIRouter(
    tags=["TrainningFirstStageForRankerInference"], prefix="/first_stage_models_for_ranker_inference"
)


# ------------------------------
# Trainning for Ranker Trainning
# ------------------------------

@trainning_first_stage_for_ranker_trainning_router.post(
    path="/knn_based_models/",
    status_code=status.HTTP_201_CREATED,
)
async def train_knn_models_for_ranker_trainning(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
) -> Response:
    """ """

    try:

        Trainner.train_all_knn_models(
            dataset=create_rectools_dataset(
                models_data=pl.scan_parquet(data_path + "base_models_data.parquet")
            ),
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )
        return Response(
            content="KNN based models trained!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while trainning KNN based models: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


@trainning_first_stage_for_ranker_trainning_router.post(
    path="/lfm_model/",
    status_code=status.HTTP_201_CREATED,
)
async def train_lfm_model_for_ranker_trainning(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
) -> Response:
    """ """

    try:

        Trainner.train_lfm_model(
            dataset=create_rectools_dataset(
                models_data=pl.scan_parquet(data_path + "base_models_data.parquet")
            ),
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )
        return Response(
            content="LFM model trained!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while trainningLFM model: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


@trainning_first_stage_for_ranker_trainning_router.post(
    path="/all_models/",
    status_code=status.HTTP_201_CREATED,
)
async def train_all_models_for_ranker_trainning(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
) -> Response:
    """ """

    try:

        Trainner.train_all_models(
            dataset=create_rectools_dataset(
                models_data=pl.scan_parquet(data_path + "base_models_data.parquet")
            ),
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

        return Response(
            content="1st stage models trained!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while trainning 1st stage models: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


# ------------------------------
# Trainning for Ranker Inference
# ------------------------------

@trainning_first_stage_for_ranker_inference_router.post(
    path="/knn_based_models/",
    status_code=status.HTTP_201_CREATED,
)
async def train_knn_models_for_ranker_inference(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
) -> Response:
    """ """

    try:

        Trainner.train_all_knn_models(
            dataset=create_rectools_dataset(
                models_data=pl.concat(
                    [
                        pl.scan_parquet(data_path + "base_models_data.parquet").select(
                            [
                                "user_id",
                                "item_id",
                                "dt",
                                "cum_weight",
                            ]
                        ),
                        pl.scan_parquet(data_path + "ranker_data.parquet").select(
                            [
                                "user_id",
                                "item_id",
                                "dt",
                                "cum_weight",
                            ]
                        ),
                    ],
                    how="vertical",
                )
            ),
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )
        return Response(
            content="KNN based models trained!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while trainning KNN based models: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


@trainning_first_stage_for_ranker_inference_router.post(
    path="/lfm_model/",
    status_code=status.HTTP_201_CREATED,
)
async def train_lfm_model_for_ranker_inference(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
) -> Response:
    """ """

    try:

        Trainner.train_lfm_model(
            dataset=create_rectools_dataset(
                models_data=pl.concat(
                    [
                        pl.scan_parquet(data_path + "base_models_data.parquet").select(
                            [
                                "user_id",
                                "item_id",
                                "dt",
                                "cum_weight",
                            ]
                        ),
                        pl.scan_parquet(data_path + "ranker_data.parquet").select(
                            [
                                "user_id",
                                "item_id",
                                "dt",
                                "cum_weight",
                            ]
                        ),
                    ],
                    how="vertical",
                )
            ),
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )
        return Response(
            content="LFM model trained!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while trainningLFM model: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


@trainning_first_stage_for_ranker_inference_router.post(
    path="/all_models/",
    status_code=status.HTTP_201_CREATED,
)
async def train_all_models_for_ranker_inference(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
) -> Response:
    """ """

    try:

        Trainner.train_all_models(
            dataset=create_rectools_dataset(
                models_data=pl.concat(
                    [
                        pl.scan_parquet(data_path + "base_models_data.parquet").select(
                            [
                                "user_id",
                                "item_id",
                                "dt",
                                "cum_weight",
                            ]
                        ),
                        pl.scan_parquet(data_path + "ranker_data.parquet").select(
                            [
                                "user_id",
                                "item_id",
                                "dt",
                                "cum_weight",
                            ]
                        ),
                    ],
                    how="vertical",
                )
            ),
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

        return Response(
            content="1st stage models trained!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while trainning 1st stage models: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )

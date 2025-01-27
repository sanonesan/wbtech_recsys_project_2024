# pylint: disable=R0801
"""
Router for preprocessing data for trainnig second stage
"""
import dill
import polars as pl

from fastapi import APIRouter, status, Response

from src.configurations.settings import SETTINGS

from src.ml.models.first_stage.inferencer import Inferencer
from src.ml.models.model_utils import create_rectools_dataset


inference_first_stage_for_ranker_trainning_router = APIRouter(
    tags=["InferenceFirstStageForRankerTrainning"],
    prefix="/first_stage_models_for_ranker_trainning",
)

inference_first_stage_for_ranker_inference_router = APIRouter(
    tags=["InferenceFirstStageForRankerInference"],
    prefix="/first_stage_models_for_ranker_inference",
)


# ------------------------------
# Inference for Ranker Trainning
# ------------------------------


@inference_first_stage_for_ranker_trainning_router.post(
    path="/knn_based_models/",
    status_code=status.HTTP_201_CREATED,
)
async def inference_knn_models_for_ranker_trainning(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
) -> Response:
    """ """

    try:
        with open(data_path + "b2r_users.dill", "rb") as f:

            Inferencer.inference_all_knn_models(
                models_path=models_path,
                candidates_data_path=candidates_data_path,
                dataset=create_rectools_dataset(
                    models_data=pl.scan_parquet(data_path + "base_models_data.parquet")
                ),
                users=dill.load(f),
                n_candidates=15,
            )

        return Response(
            content="KNN based models inferenced!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while inferencing KNN based models: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


@inference_first_stage_for_ranker_trainning_router.post(
    path="/lfm_model/",
    status_code=status.HTTP_201_CREATED,
)
async def inference_lfm_model_for_ranker_trainning(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
) -> Response:
    """ """

    try:

        with open(data_path + "b2r_users.dill", "rb") as f:

            Inferencer.inference_lfm_model(
                models_path=models_path,
                candidates_data_path=candidates_data_path,
                dataset=create_rectools_dataset(
                    models_data=pl.scan_parquet(data_path + "base_models_data.parquet")
                ),
                users=dill.load(f),
                n_candidates=15,
            )

        return Response(
            content="LFM model inferenced!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while inferencing LFM model: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


@inference_first_stage_for_ranker_trainning_router.post(
    path="/all_models/",
    status_code=status.HTTP_201_CREATED,
)
async def inference_all_models_for_ranker_trainning(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
) -> Response:
    """ """

    try:

        with open(data_path + "b2r_users.dill", "rb") as f:

            Inferencer.inference_all_models(
                models_path=models_path,
                candidates_data_path=candidates_data_path,
                dataset=create_rectools_dataset(
                    models_data=pl.scan_parquet(data_path + "base_models_data.parquet")
                ),
                users=dill.load(f),
                n_candidates=15,
            )

        return Response(
            content="1st stage models inferenced!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while inferencing 1st stage models: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


# ------------------------------
# Inference for Ranker Inference
# ------------------------------


@inference_first_stage_for_ranker_inference_router.post(
    path="/knn_based_models/",
    status_code=status.HTTP_201_CREATED,
)
async def inference_knn_models_for_ranker_inference(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
) -> Response:
    """ """

    try:

        with open(data_path + "bNr2t_users.dill", "rb") as f:

            Inferencer.inference_all_knn_models(
                models_path=models_path,
                candidates_data_path=candidates_data_path,
                dataset=create_rectools_dataset(
                    models_data=pl.concat(
                        [
                            pl.scan_parquet(
                                data_path + "base_models_data.parquet"
                            ).select(
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
                users=dill.load(f),
                n_candidates=15,
            )

        return Response(
            content="KNN based models inferenced!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while inferencing KNN based models: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


@inference_first_stage_for_ranker_inference_router.post(
    path="/lfm_model/",
    status_code=status.HTTP_201_CREATED,
)
async def inference_lfm_model_for_ranker_inference(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
) -> Response:
    """ """

    try:

        with open(data_path + "bNr2t_users.dill", "rb") as f:

            Inferencer.inference_lfm_model(
                models_path=models_path,
                candidates_data_path=candidates_data_path,
                dataset=create_rectools_dataset(
                    models_data=pl.concat(
                        [
                            pl.scan_parquet(
                                data_path + "base_models_data.parquet"
                            ).select(
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
                users=dill.load(f),
                n_candidates=15,
            )

        return Response(
            content="LFM model inferenced!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while inferencing LFM model: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )


@inference_first_stage_for_ranker_inference_router.post(
    path="/all_models/",
    status_code=status.HTTP_201_CREATED,
)
async def inference_all_models_for_ranker_inference(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
) -> Response:
    """ """

    try:

        with open(data_path + "bNr2t_users.dill", "rb") as f:

            Inferencer.inference_all_models(
                models_path=models_path,
                candidates_data_path=candidates_data_path,
                dataset=create_rectools_dataset(
                    models_data=pl.concat(
                        [
                            pl.scan_parquet(
                                data_path + "base_models_data.parquet"
                            ).select(
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
                users=dill.load(f),
                n_candidates=15,
            )

        return Response(
            content="1st stage models inferenced!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while inferencing 1st stage models: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )

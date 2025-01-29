# pylint: disable=R0801
"""
Router for preprocessing data for trainnig cold recs
"""

import polars as pl

from fastapi import APIRouter, status, Response

from src.configurations.settings import SETTINGS

from src.ml.models.cold_recs.trainner import Trainner

trainning_cold_recs_models_router = APIRouter(
    tags=["TrainningColdRecsModels"], prefix="/cold_recs_models"
)


@trainning_cold_recs_models_router.post(
    path="/popular_model/",
    status_code=status.HTTP_201_CREATED,
)
async def train_popular_model(
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

        Trainner.train_popular_model(
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


@trainning_cold_recs_models_router.post(
    path="/mab_model/fit",
    status_code=status.HTTP_201_CREATED,
)
async def fit_mab_model(
    data_path=SETTINGS.data_path,
    models_path=SETTINGS.models_path,
    candidates_data_path=SETTINGS.candidates_data_path,
    model_name: str = "mab",
) -> Response:
    """
    Endpoint to train the Multi-Armed Bandit (MAB) model.

    Args:
        data_path: Path to the directory containing the training data (default from SETTINGS).
        models_path: Path to the directory where trained models are saved (default from SETTINGS).
        candidates_data_path: Path to the directory to save generated recommendations 
            (default from SETTINGS).
        model_name: The base name of the model file (default is "mab").

    Returns:
        A Response object indicating success or failure of the training process, along with an
        appropriate HTTP status code.
    """
    try:

        Trainner.fit_mab_model(
            dataset=(
                pl.concat(
                    [
                        pl.scan_parquet(data_path + "base_models_data.parquet").select(
                            [
                                "user_id",
                                "item_id",
                                "dt",
                                "ui_inter",
                                "u_total_inter",
                            ]
                        ),
                        pl.scan_parquet(data_path + "ranker_data.parquet").select(
                            [
                                "user_id",
                                "item_id",
                                "dt",
                                "ui_inter",
                                "u_total_inter",
                            ]
                        ),
                    ],
                    how="vertical",
                )
                .filter(
                    (
                        (
                            pl.col("item_id").is_in(
                                pl.scan_parquet(
                                    candidates_data_path + "candidates_pop.parquet"
                                )
                                .select("item_id")
                                .collect()
                                .to_numpy()
                                .flatten()
                            )
                        )
                    )
                )
                .with_columns(
                    (pl.col("ui_inter") > 2).cast(pl.UInt8).alias("binary_weight"),
                )
                .select(
                    "user_id",
                    "item_id",
                    "dt",
                    "binary_weight",
                )
            ),
            models_path=models_path,
            model_name=model_name,
        )

        return Response(
            content="MAB model trained!",
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            content=f"Error while trainning MAB model: {e}",
            status_code=status.HTTP_409_CONFLICT,
        )

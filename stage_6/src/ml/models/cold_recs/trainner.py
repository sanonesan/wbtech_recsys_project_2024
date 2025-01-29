# pylint: disable=R0801
"""
Trainner for cold recs models
"""

from datetime import timedelta

import dill
import polars as pl


from mab2rec import BanditRecommender, LearningPolicy


from src.ml.models.model_utils import create_rectools_dataset
from src.ml.models.cold_recs.popular_model import PopModel


from src.logs.console_logger import LOGGER


class Trainner:
    """
    Trainner for cold recs models
    """

    @staticmethod
    def fit_mab_model(
        dataset: pl.LazyFrame,
        models_path: str,
        model_name: str = "mab",
        time_delta: timedelta = timedelta(hours=4),
    ):
        """
        Fits a Multi-Armed Bandit (MAB) model to a dataset over time.

        This method iteratively fits a BanditRecommender model using Thompson Sampling
        to training data divided into time chunks, until the entire training data period is covered.
        The model is then saved to disk.

        Args:
            dataset: A Polars LazyFrame containing training data with columns 'dt' (datetime)
                , 'item_id' (integer), and 'binary_weight' (numeric).
            models_path: The path to the directory where the fitted MAB model will be saved.
            model_name: The base name of the model file (default is 'mab').
            time_delta: The time interval defining chunks of training data for fitting
                (default is 4 hours).

        Returns:
            None
        """

        mab_model = BanditRecommender(
            LearningPolicy.ThompsonSampling(),
            top_k=15,
            n_jobs=-1,
        )

        first_date = dataset.select("dt").min().collect().item()
        last_date = dataset.select("dt").max().collect().item()

        right = first_date
        left, right = right, right + time_delta

        chunk = dataset.filter(
            pl.col("dt").is_between(left, right, closed="left")
        ).collect()

        mab_model.fit(
            decisions=chunk["item_id"].to_numpy(),
            rewards=chunk["binary_weight"].to_numpy(),
        )
        LOGGER.info(f"MAB model fitted: {left}, {right}")

        while right <= last_date:

            left, right = right, right + time_delta

            chunk = dataset.filter(
                pl.col("dt").is_between(left, right, closed="left")
            ).collect()

            mab_model.partial_fit(
                decisions=chunk["item_id"].to_numpy(),
                rewards=chunk["binary_weight"].to_numpy(),
            )

            LOGGER.info(f"MAB model fitted: {left}, {right}")

        # Save model
        with open(models_path + f"{model_name}.dill", "wb") as f:
            dill.dump(mab_model, f)
            LOGGER.info("MAB model: dumped!")

    @staticmethod
    def partial_fit_mab_model(
        dataset: pl.LazyFrame,
        models_path: str,
        model_name: str = "mab",
        time_delta: timedelta = timedelta(hours=4),
    ):
        """
        Partially fits an existing Multi-Armed Bandit (MAB) model to new data over time.

        This method loads a pre-trained BanditRecommender model from disk, then iteratively
        updates the model with new training data divided into time chunks. This is used for
        incremental training of the model when new data becomes available without retraining
        from scratch. The model is then saved back to disk.

        Args:
            dataset: A Polars LazyFrame containing new training data with columns 'dt' (datetime)
                , 'item_id' (integer), and 'binary_weight' (numeric).
            models_path: The path to the directory where the pre-trained MAB model is stored,
                and where the updated model will be saved.
            model_name: The base name of the model file (default is 'mab').
            time_delta: The time interval defining chunks of training data for updating
                the model (default is 4 hours).

        Returns:
            None
        """

        LOGGER.info("MAB model: loading...")
        with open(models_path + f"{model_name}.dill", "rb") as f:
            mab_model: BanditRecommender = dill.load(f)
            LOGGER.info("MAB model: loaded!")

        first_date = dataset.select("dt").min().collect().item()
        last_date = dataset.select("dt").max().collect().item()

        right = first_date

        while right <= last_date:

            left, right = right, right + time_delta

            chunk = dataset.filter(
                pl.col("dt").is_between(left, right, closed="left")
            ).collect()

            mab_model.partial_fit(
                decisions=chunk["item_id"].to_numpy(),
                rewards=chunk["binary_weight"].to_numpy(),
            )

            LOGGER.info(f"MAB model fitted: {left}, {right}")

        # Save model
        with open(models_path + f"{model_name}.dill", "wb") as f:
            dill.dump(mab_model, f)
            LOGGER.info("MAB model: dumped!")

    @staticmethod
    def train_popular_model(
        data_path: str,
        models_path: str,
        candidates_data_path: str,
    ):
        """
        Trains a Popular model for cold recs.

        This static method initializes and fits an Popular model using
        the provided dataset. It logs the start and end
        of the initialization and fitting process.

        Args:
            dataset (RTDataset): Input rectools Dataset used for training.
            models_path (str): Path to the directory to store the trained model.
            candidates_data_path (str): Path to the candidate data used for the model.
        """

        LOGGER.info(msg="Trainning 1st stage LFM model: started...")

        LOGGER.info(msg="Popular model initialization: started...")
        model = PopModel(
            models_path=models_path,
            model_name="pop",
            candidates_data_path=candidates_data_path,
            fitted=False,
        )
        LOGGER.info(msg="Popular model initialization: finished!")

        LOGGER.info(msg="Popular model fitting: started...")
        model.fit(
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
            )
        )
        LOGGER.info(msg="Popular model fitting: finished!")

        LOGGER.info(msg="Trainning Popular model: finished!")

# pylint: disable=R0801
"""
Inferencer for cold recs model
"""

import dill
import polars as pl

from src.ml.models.model_utils import create_rectools_dataset
from src.ml.models.cold_recs.popular_bandit import PopularBanditRecommender
from src.ml.models.cold_recs.popular_model import PopModel

from src.logs.console_logger import LOGGER


class Inferencer:  # pylint: disable=R0903
    """
    Inferencer for cold recs model
    """

    @staticmethod
    def inference_popular_bandit_model(  # pylint: disable=R0913 R0917
        data_path: str,
        models_path: str,
        candidates_data_path: str,
        candidates_file_name: str = "candidates_mab",
    ):
        """
        Generates recommendations using a PopularBanditRecommender model.

        This static method initializes a PopularBanditRecommender, loads pre-trained
        MAB and popular models, then generates recommendations for given users using
        a combination of popularity and bandit predictions. Finally, saves result to parquet.

        Args:
            users: A NumPy array-like of user IDs for whom recommendations are to be generated.
            models_path: Path to the directory where the pre-trained 'mab.dill' and
                        'pop.dill' models are stored.
            dataset: An AbstractDataset containing the training data that was used for models.
            candidates_data_path: Path to the directory where the generated candidates will be
                        saved.
            candidates_file_name: The base name of the candidates file to be saved
                                (default is "candidates_mab").

        Returns:
            None
        """

        LOGGER.info("Init pop_bandit_model: started...")

        pop_bandit_model = PopularBanditRecommender(
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
            path_bandit_model=models_path + "mab.dill",
            path_popular_model=models_path + "pop.dill",
            top_k=15,
        )

        LOGGER.info("Init pop_bandit_model: finished!")

        LOGGER.info("Inference pop_bandit_model: started...")

        with open(data_path + "test_only_users.dill", "rb") as f:
            test_only_users = dill.load(f)

        pl.from_pandas(
            pop_bandit_model.predict(
                test_only_users,
                pop_k=50,
                pre_gen_recs=True,
                pre_gen_n=100,
            )
        ).write_parquet(candidates_data_path + f"{candidates_file_name}.parquet")

        LOGGER.info("Inference pop_bandit_model: finished!")

    @staticmethod
    def inference_popular_model(  # pylint: disable=R0913 R0917
        data_path: str,
        models_path: str,
        candidates_data_path: str,
    ):
        """
        Generates recommendations using a Popular model.

        Args:
            data_path: Path to the directory with all the data for trainning and inference.
            models_path: Path to the directory where the pre-trained 'mab.dill' and
                        'pop.dill' models are stored.
            candidates_data_path: Path to the directory where the generated candidates will be
                        saved.
        Returns:
            None
        """

        LOGGER.info("Init popular model: started...")

        popular_model = PopModel(
            models_path=models_path,
            model_name="pop",
            candidates_data_path=candidates_data_path,
            fitted=True,
        )

        LOGGER.info("Init pop_bandit_model: finished!")

        LOGGER.info("Inference pop_bandit_model: started...")

        with open(data_path + "test_only_users.dill", "rb") as f:
            test_only_users = dill.load(f)

        popular_model.get_candidates(
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
            users=test_only_users[0],
            n_candidates=1000,
        )

        LOGGER.info("Inference pop_bandit_model: finished!")

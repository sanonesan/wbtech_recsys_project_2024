# pylint: disable=R0801
"""
Inferencer for ranker model
"""

from src.ml.models.second_stage.ranker_params import CB_PARAMS
from src.ml.models.second_stage.catboost_ranker import CBRanker

from src.logs.console_logger import LOGGER


class Inferencer:  # pylint: disable=R0903
    """
    Inferencer for ranker model
    """

    @staticmethod
    def inference_catboost_model(  # pylint: disable=R0913 R0917
        data_path: str,
        ranker_data: str,
        models_path: str,
        candidates_data_path: str,
        model_name: str = "cb_ranker",
        n_splits: int = 1,
    ):
        """
        Performs inference using a trained CatBoost ranking model.

        Args:
            data_path (str): Base path to the inference data.
            ranker_data (str): Path to the specific ranking data. This path will be added
                to the `data_path`.
            models_path (str): Path to the directory where the trained model is saved.
            candidates_data_path (str): Path to the candidate data used for ranking.
            model_name (str, optional): Name of the model to load. Defaults to 'cb_ranker'.
            n_splits (int, optional): Number of splits for parallel processing or inference.
                Defaults to 1.

        Returns:
            Any: The rearranged candidates returned
                by the `cb_ranker_model.get_rearranged_candidates()` method.
        """

        LOGGER.info("Init catboost ranker: started...")
        cb_ranker_model = CBRanker(
            models_path=models_path,
            model_name=model_name,
            candidates_data_path=candidates_data_path,
            load_model=True,
            **CB_PARAMS,
        )
        LOGGER.info("Init catboost ranker: finished!")

        LOGGER.info(f"{cb_ranker_model.get_params()}")

        LOGGER.info("Inference catboost ranker: started...")
        cb_ranker_model.get_rearranged_candidates(
            data_path + ranker_data,
            n_splits=n_splits,
        )
        LOGGER.info("Inference catboost ranker: finished!")

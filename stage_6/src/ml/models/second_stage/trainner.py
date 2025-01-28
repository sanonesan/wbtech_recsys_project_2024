# pylint: disable=R0801
"""
Trainner for ranker model
"""

from src.ml.models.second_stage.ranker_params import CB_PARAMS
from src.ml.models.second_stage.catboost_ranker import CBRanker

from src.logs.console_logger import LOGGER


class Trainner:  # pylint: disable=R0903
    """
    Trainner for ranker model
    """

    @staticmethod
    def train_catboost_model(
        data_path: str,
        models_path: str,
        candidates_data_path: str,
        model_name: str = "cb_ranker",
    ):
        """
        Trains a CatBoost ranking model.

        Args:
            data_path (str): Path to the training data.
            models_path (str): Path where the trained model will be saved.
            candidates_data_path (str): Path to the candidates data used in the model.
            model_name (str, optional): Name of the model, defaults to 'cb_ranker'.

        Returns:
            None
        """
        LOGGER.info("Init catboost ranker: started...")
        cb_ranker_model = CBRanker(
            models_path=models_path,
            model_name=model_name,
            candidates_data_path=candidates_data_path,
            load_model=False,
            **CB_PARAMS,
        )
        LOGGER.info("Init catboost ranker: finished!")

        LOGGER.info(f"{cb_ranker_model.get_params()}")

        LOGGER.info("Trainning catboost ranker: started...")
        cb_ranker_model.my_fit(data_path=data_path)
        LOGGER.info("Trainning catboost ranker: finished!")

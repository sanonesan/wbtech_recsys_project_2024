# pylint: disable=R0801
"""
Trainner for first stage models
"""

from rectools.dataset import Dataset as RTDataset

from src.ml.models.first_stage.knn_based_models import KNNBasedModel
from src.ml.models.first_stage.lfm_model import LFMModel

from src.logs.console_logger import LOGGER


class Trainner:
    """
    Trainner for first stage models
    """

    @staticmethod
    def train_all_knn_models(
        dataset: RTDataset,
        models_path: str,
        candidates_data_path: str,
    ):
        """
        Trains all specified KNN-based first-stage models.

        This static method iterates through different KNN-based model types
        (CosineRecommender, BM25Recommender, TFIDFRecommender), initializes each model,
        fits it to the provided dataset, and saves the trained model.

        Args:
            dataset (RTDataset): Input rectools Dataset used for training.
            models_path (str): Directory to store the trained models.
            candidates_data_path (str): Path to the candidate data used for the models.
        """

        model_type_name_dict = {
            "CosineRecommender": "cos",
            "BM25Recommender": "bm25",
            "TFIDFRecommender": "tfidf",
        }

        LOGGER.info(msg="Trainning 1st stage KNN models: started...")

        for model_type in ["CosineRecommender", "BM25Recommender", "TFIDFRecommender"]:

            LOGGER.info(msg=f"{model_type} models initialization: started...")
            model = KNNBasedModel(
                models_path=models_path,
                model_name=f"{model_type_name_dict[model_type]}",
                candidates_data_path=candidates_data_path,
                model_type=model_type,
            )
            LOGGER.info(msg=f"{model_type} models initialization: finished!")

            LOGGER.info(msg=f"{model_type} models fitting: started...")
            model.fit(dataset=dataset)
            LOGGER.info(msg=f"{model_type} models fitting: finished!")

        LOGGER.info(msg="Trainning 1st stage KNN models: finished!")

    @staticmethod
    def train_lfm_model(
        dataset: RTDataset,
        models_path: str,
        candidates_data_path: str,
    ):
        """
        Trains a LFM for the first stage.

        This static method initializes and fits an LFM model using the provided dataset.
        It logs the start and end of the initialization and fitting process.

        Args:
            dataset (RTDataset): Input rectools Dataset used for training.
            models_path (str): Path to the directory to store the trained model.
            candidates_data_path (str): Path to the candidate data used for the model.
        """

        LOGGER.info(msg="Trainning 1st stage LFM model: started...")

        LOGGER.info(msg="LFM model initialization: started...")
        model = LFMModel(
            models_path=models_path,
            model_name="lfm",
            candidates_data_path=candidates_data_path,
            fitted=False,
        )
        LOGGER.info(msg="LFM model initialization: finished!")

        LOGGER.info(msg="LFM model fitting: started...")
        model.fit(dataset=dataset)
        LOGGER.info(msg="LFM model fitting: finished!")

        LOGGER.info(msg="Trainning 1st stage LFM model: finished!")

    @staticmethod
    def train_all_models(
        dataset: RTDataset,
        models_path: str,
        candidates_data_path: str,
    ):
        """
        Trains all first stage models.

        Args:
            dataset (RTDataset): Input rectools Dataset used for training.
            models_path (str): Path to the directory to store the trained model.
            candidates_data_path (str): Path to the candidate data used for the model.
        """

        LOGGER.info(msg="Trainning 1st stage models: started...")

        Trainner.train_all_knn_models(
            dataset=dataset,
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

        Trainner.train_lfm_model(
            dataset=dataset,
            models_path=models_path,
            candidates_data_path=candidates_data_path,
        )

        LOGGER.info(msg="Trainning 1st stage models: finished!")

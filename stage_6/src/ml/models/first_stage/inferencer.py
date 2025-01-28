# pylint: disable=R0801
"""
Inferencer for 1st stage models.
"""

import numpy.typing as npt
from rectools.dataset import Dataset as RTDataset

from src.ml.models.first_stage.knn_based_models import KNNBasedModel
from src.ml.models.first_stage.lfm_model import LFMModel

from src.logs.console_logger import LOGGER


class Inferencer:
    """
    Class for performing inference with 1st stage models.
    """

    @staticmethod
    def inference_all_knn_models(
        models_path: str,
        candidates_data_path: str,
        dataset: RTDataset,
        users: npt.ArrayLike,
        n_candidates: int,
    ):
        """
        Performs inference using all specified KNN-based first-stage models.

        This static method iterates through different KNN-based model types
        (CosineRecommender, BM25Recommender, TFIDFRecommender), initializes each model,
        and generates recommendations using `get_candidates` function.

        Args:
            models_path (str): Directory where trained models are stored.
            candidates_data_path (str): Path to the candidate data.
            dataset (RTDataset): Input rectools Dataset.
            users (npt.NDArray): Array of user IDs for which to generate recommendations.
            n_candidates (int): Number of candidate items to recommend for each user.
        """

        model_type_name_dict = {
            "CosineRecommender": "cos",
            "BM25Recommender": "bm25",
            "TFIDFRecommender": "tfidf",
        }

        LOGGER.info(msg="Inferencing 1st stage KNN models: started...")

        for model_type in ["CosineRecommender", "BM25Recommender", "TFIDFRecommender"]:

            LOGGER.info(msg=f"{model_type} models initialization: started...")
            model = KNNBasedModel(
                models_path=models_path,
                model_name=f"{model_type_name_dict[model_type]}",
                candidates_data_path=candidates_data_path,
                model_type=model_type,
                fitted=True,
            )
            LOGGER.info(msg=f"{model_type} models initialization: finished!")

            LOGGER.info(msg=f"{model_type} models inference: started...")
            model.get_candidates(
                dataset=dataset,
                users=users,
                n_candidates=n_candidates,
            )
            LOGGER.info(msg=f"{model_type} models inference: finished!")

        LOGGER.info(msg="Inferencing 1st stage KNN models: finished!")

    @staticmethod
    def inference_lfm_model(
        models_path: str,
        candidates_data_path: str,
        dataset: RTDataset,
        users: npt.ArrayLike,
        n_candidates: int,
    ):
        """
        Performs inference using a trained LFM model.

        This static method initializes an LFM model, loads it from disk, and generates
        recommendations using `get_candidates` method.

        Args:
            models_path (str): Directory to the stored models.
            candidates_data_path (str): Path to the candidate data.
            dataset (RTDataset): Input rectools Dataset to generate recommendations.
            users (npt.NDArray): Array-like object of user IDs 
                for which to generate recommendations.
            n_candidates (int): The number of candidate items to recommend for each user.
        """

        LOGGER.info(msg="Inferencing 1st stage LFM model: started...")

        LOGGER.info(msg="LFM model initialization: started...")
        model = LFMModel(
            models_path=models_path,
            model_name="lfm",
            candidates_data_path=candidates_data_path,
            fitted=True,
        )
        LOGGER.info(msg="LFM model initialization: finished!")

        LOGGER.info(msg="LFM model inference: started...")
        model.get_candidates(
            dataset=dataset,
            users=users,
            n_candidates=n_candidates,
        )
        LOGGER.info(msg="LFM model inference: finished!")

        LOGGER.info(msg="Inferencing 1st stage LFM model: finished!")

    @staticmethod
    def inference_all_models(
        models_path: str,
        candidates_data_path: str,
        dataset: RTDataset,
        users: npt.ArrayLike,
        n_candidates: int,
    ):
        """
        Performs inference using all first-stage models (KNN and LFM).

        This static method calls the inference methods for KNN-based models and LFM model,
        in order to generate recommendations.

        Args:
            models_path (str): Directory where the models are stored.
            candidates_data_path (str): Path where the generated candidates are stored.
            dataset (RTDataset): Input rectools Dataset used to generate the recommendations.
            users (npt.NDArray): Array-like of user IDs for whom recommendations are generated.
            n_candidates (int): The number of candidate items to recommend for each user.
        """

        LOGGER.info(msg="Inferencing 1st stage models: started...")

        Inferencer.inference_all_knn_models(
            models_path=models_path,
            candidates_data_path=candidates_data_path,
            dataset=dataset,
            users=users,
            n_candidates=n_candidates,
        )

        Inferencer.inference_lfm_model(
            models_path=models_path,
            candidates_data_path=candidates_data_path,
            dataset=dataset,
            users=users,
            n_candidates=n_candidates,
        )

        LOGGER.info(msg="Inferencing 1st stage models: finished!")

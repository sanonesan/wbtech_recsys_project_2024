"""
KNN implementation using implicit (custom wrapper)
"""

import dill

from rectools.dataset import Dataset as RTDataset
from rectools.models import (
    implicit_knn,
)
from implicit import nearest_neighbours

from src.ml.models.first_stage.base_rt_model import BaseRTModel


class KNNBasedModel(BaseRTModel):  # pylint: disable=R0903
    """
    Class wrapper for KNN implementation using implicit
    """

    model_path: str
    model_name: str
    candidates_data_path: str
    model_type: str
    k: int
    fitted: bool

    def __init__(  # pylint: disable=R0913, R0917
        self,
        models_path,
        model_name,
        candidates_data_path,
        fitted: bool = False,
        model_type: str = "CosineRecommender",
        k: int = 50,
    ):
        """
        Initializes a KNN-based recommender model.

        Args:
            models_path (str): Path where the models are stored.
            model_name (str): Name of the model.
            candidates_data_path (str): Path where to store the candidates data.
            fitted (bool, optional): Flag indicating if the model is already fitted,
                defaults to False.
            model_type (str, optional): Type of model ('CosineRecommender',
                'BM25Recommender', 'TFIDFRecommender'), defaults to 'CosineRecommender'.
            K (int, optional): Number of nearest neighbors to consider. Defaults to 50.
        """
        super().__init__(
            models_path=models_path,
            model_name=model_name,
            candidates_data_path=candidates_data_path,
            fitted=fitted,
        )
        self.model_type = model_type
        self.k = k

    def fit(self, dataset: RTDataset):
        """
        Fits the KNN-based model to the dataset.

        This method initializes the corresponding KNN recommender model based on
        `self.model_type`, fits it using the input `dataset`, and saves the trained model.

        Args:
            dataset (RTDataset): The input rectools dataset used for training the model.
        """

        with open(self.model_path, "wb") as f:

            match self.model_type:
                case "CosineRecommender":
                    model = implicit_knn.ImplicitItemKNNWrapperModel(
                        model=nearest_neighbours.CosineRecommender(K=self.k)
                    )

                case "BM25Recommender":
                    model = implicit_knn.ImplicitItemKNNWrapperModel(
                        model=nearest_neighbours.BM25Recommender(K=self.k)
                    )

                case "TFIDFRecommender":
                    model = implicit_knn.ImplicitItemKNNWrapperModel(
                        model=nearest_neighbours.TFIDFRecommender(K=self.k)
                    )

            model.fit(dataset)

            # save model
            dill.dump(model, f)

        self.fitted = True

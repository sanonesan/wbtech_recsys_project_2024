"""
Abstract class for rectools models (custom wrapper)
"""

import dill
import numpy.typing as npt
import polars as pl
from rectools.dataset import Dataset as RTDataset

from src.ml.models.base_models.base_model import BaseModel


class BaseRTModel(BaseModel):
    """
    Abstract base class for Rectools models (custom wrapper).

    This class defines the basic structure and interface for all Rectools-based
    models used in the project. It includes properties for model name, path,
    candidate data path, and a flag to indicate if the model has been fitted.
    """

    def fit(self, dataset: RTDataset):  # pylint: disable=W0613 W0221
        """
        Abstract method to fit the model.

        This method should be implemented by subclasses to perform model fitting.

        Args:
            dataset (RTDataset): The input rectools Dataset used for training.
        """

    def get_candidates(  # pylint: disable=W0613 W0221 W0221
        self,
        dataset: RTDataset,
        users: npt.ArrayLike,
        n_candidates: npt.ArrayLike,
    ) -> None:
        """
        Generates and saves candidate recommendations using a fitted model.

        This method loads a fitted model, generates recommendations for the specified
        users, adds a model-specific score and rank, and saves the results to a parquet file.

        Args:
            dataset (RTDataset): Rectools dataset used for generating recommendations.
            users (npt.NDArray): Array-like of user IDs for which to generate recommendations.
            n_candidates (int): Number of candidate items to recommend for each user.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if self.fitted:
            with open(self.model_path, "rb") as f:
                pl.from_pandas(
                    dill.load(f).recommend(
                        users,
                        dataset,
                        # выдаем n_candidates кандидатов
                        k=n_candidates,
                        # рекомендуем уже просмотренные товары
                        filter_viewed=False,
                    )
                ).rename(
                    {
                        "score": f"{self.model_name}_score",
                        "rank": f"{self.model_name}_rank",
                    }
                ).write_parquet(
                    self.candidates_data_path + f"candidates_{self.model_name}.parquet"
                )

        else:
            raise LookupError("Model is not fitted")

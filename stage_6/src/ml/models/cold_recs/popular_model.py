"""
Popular model implementation using rectools (custom wrapper)
"""

import dill

from rectools.dataset import Dataset as RTDataset
from rectools.models import (
    PopularModel,
)

from src.ml.models.base_models.base_rt_model import BaseRTModel

from src.logs.console_logger import LOGGER


class PopModel(BaseRTModel):  # pylint: disable=R0903
    """
    Class custom wrapper for rectools PopularModel
    """

    def fit(self, dataset: RTDataset):
        """
        Fits the Popular Model to the given dataset.

        This method initializes model using default parameters,
        fits the model using PopularModel, and saves the trained model.

        Args:
            dataset (RTDataset): The input rectools Dataset for training the popular model.
        """
        # init
        popular_model = PopularModel()
        # fit
        popular_model.fit(dataset)

        # Save model
        with open(self.model_path, "wb") as f:
            dill.dump(popular_model, f)
            LOGGER.info("Popular model dumped!")

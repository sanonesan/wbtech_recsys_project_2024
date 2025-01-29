"""
LFM implementation using LightFM (custom wrapper)
"""

import dill

from lightfm import LightFM

from rectools.dataset import Dataset as RTDataset
from rectools.models import LightFMWrapperModel

from src.ml.models.base_models.base_rt_model import BaseRTModel


class LFMModel(BaseRTModel):  # pylint: disable=R0903
    """
    Latent Factor Model (LFM) implementation using LightFM.

    This class extends BaseRTModel and provides a specific implementation for
    training and saving a LightFM model using the LightFMWrapperModel class.
    """

    def fit(self, dataset: RTDataset):
        """
        Fits the Latent Factor Model (LFM) to the given dataset.

        This method initializes a LightFM model using default parameters,
        fits the model using LightFMWrapperModel, and saves the trained model.

        Args:
            dataset (RTDataset): The input rectools Dataset for training the LFM model.
        """
        # init
        lfm_model = LightFMWrapperModel(
            LightFM(
                no_components=64,
                learning_rate=0.1,
                loss="warp",
                max_sampled=7,
            ),
            epochs=20,
            num_threads=6,
            verbose=1,
        )
        # fit
        lfm_model.fit(dataset)

        # Save model
        with open(self.model_path, "wb") as f:
            dill.dump(lfm_model, f)

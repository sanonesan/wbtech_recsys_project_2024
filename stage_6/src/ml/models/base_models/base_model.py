"""
Abstract class for rectools models (custom wrapper)
"""

from abc import ABC
from dataclasses import dataclass

@dataclass
class BaseModel(ABC):
    """
    Abstract base class for Rectools models (custom wrapper).

    This class defines the basic structure and interface for all Rectools-based
    models used in the project. It includes properties for model name, path,
    candidate data path, and a flag to indicate if the model has been fitted.
    """

    def __init__(
        self,
        models_path,
        model_name,
        candidates_data_path,
        fitted: bool = False,
    ):
        self.model_name = model_name
        self.model_path = models_path + "/" + model_name + ".dill"
        self.candidates_data_path = candidates_data_path

        # Params for trainning and inferencing model
        self.fitted = fitted

    def fit(self):
        """
        Abstract method to fit the model.
        """

    def get_candidates(self):
        """
        Generates and saves candidate recommendations using a fitted model.
        """

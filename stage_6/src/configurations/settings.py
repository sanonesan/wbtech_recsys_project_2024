"""
Default settings for database connections and paths to data
"""

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Class with default settings for application
    """
    project_name: str = "WBTech proj"

    interactions_path: str
    text_data_path: str
    imgs_data_path: str
    data_path: str 
    models_path: str 
    candidates_data_path: str
    do_ranker_test: bool
    imgs_batch_size: int


load_dotenv()
SETTINGS = Settings(
    interactions_path=os.getenv("INTERACTIONS_PATH"),
    text_data_path=os.getenv("TEXT_DATA_PATH"),
    imgs_data_path=os.getenv("IMGS_DATA_PATH"),
    data_path=os.getenv("DATA_PATH"),
    models_path=os.getenv("MODELS_PATH"),
    candidates_data_path=os.getenv("CANDIDATES_DATA_PATH"),
    do_ranker_test=False,
    imgs_batch_size=512,
)

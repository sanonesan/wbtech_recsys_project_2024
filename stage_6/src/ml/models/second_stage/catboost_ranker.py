"""
CatBoost ranker model custom wrapper
"""

import numpy as np
import polars as pl

from tqdm.auto import tqdm

from catboost import CatBoostRanker, Pool

from src.ml.data.feature_utils import (
    SECOND_STAGE_FEATURES,
    SECOND_STAGE_CATEGORIAL_FEATURES,
    FEATURES_FOR_ANALYSIS,
)
from src.ml.models.model_utils import (
    get_catboost_group_id,
    add_score_and_rank,
)

RANDOM_STATE = 42


class CBRanker(CatBoostRanker):
    """
    CatBoost ranker model custom wrapper
    """

    def __init__(
        self,
        models_path: str,
        model_name: str,
        candidates_data_path: str,
        load_model: bool = False,
        **params,
    ):
        """
        Initializes the CBRanker model.

        Args:
            models_path (str): Path to the directory where trained models are stored.
            model_name (str): Name of the model. This name will be used to save and load the model.
            candidates_data_path (str): Path to the candidates data used for ranking.
            load_model (bool, optional): If True, loads a model from the model_path. 
                Defaults to False.
            **params (Dict[str, Any]): Additional parameters to pass 
                to the CatBoostRanker constructor.
        """

        super().__init__(**params)

        self.model_name = model_name
        self.model_path = models_path + "/" + model_name + ".cbm"
        self.candidates_data_path = candidates_data_path

        if load_model:
            self.load_model(self.model_path)

    def my_fit(self, data_path: str):
        """
        Trains the CatBoost ranking model.

        This method reads training and validation data from parquet files,
        creates CatBoost Pools, and trains the model using these pools. It then saves the
        trained model to the specified path.
        Error handling is implemented using a try-except block.

        Args:
            data_path (str): Base path to the training and validation data files.
                            It assumes the files are named "ranker_train.parquet" and
                            "ranker_val.parquet" and are located in the specified path.
        """

        ranker_train = pl.scan_parquet(data_path + "ranker_train.parquet").sort(
            by="user_id"
        )
        ranker_val = pl.scan_parquet(data_path + "ranker_val.parquet").sort(
            by="user_id"
        )

        print(
            Pool(
                data=ranker_train.select(SECOND_STAGE_FEATURES).collect().to_pandas(),
                label=ranker_train.select("target").collect().to_pandas(),
                group_id=get_catboost_group_id(ranker_train),
                cat_features=SECOND_STAGE_CATEGORIAL_FEATURES,
            )
        )
        try:
            super().fit(
                # Train Pool
                Pool(
                    data=ranker_train.select(SECOND_STAGE_FEATURES)
                    .collect()
                    .to_pandas(),
                    label=ranker_train.select("target").collect().to_pandas(),
                    group_id=get_catboost_group_id(ranker_train),
                    cat_features=SECOND_STAGE_CATEGORIAL_FEATURES,
                ),
                # Val Pool
                eval_set=Pool(
                    data=ranker_val.select(SECOND_STAGE_FEATURES).collect().to_pandas(),
                    label=ranker_val.select("target").collect().to_pandas(),
                    group_id=get_catboost_group_id(ranker_val),
                    cat_features=SECOND_STAGE_CATEGORIAL_FEATURES,
                ),
            )
        except Exception as e:
            print(e)

        # Save Model
        self.save_model(self.model_path)

    def get_rearranged_candidates(self, ranker_data_path: str, n_splits: int = 1):
        """
        Generates and saves rearranged candidate rankings using a trained CatBoost model.

        This method reads ranking data, splits it into batches (if n_splits > 1),
        performs predictions using the CatBoost model, adds prediction scores and ranks to
        the data, and saves the results to parquet files.


        Args:
            ranker_data_path (str): Path to the ranking data (parquet file).
            n_splits (int, optional): Number of splits for parallel processing. Defaults to 1.
        """

        ranker_data = pl.scan_parquet(ranker_data_path).sort(by="user_id")

        if n_splits > 1:
            batches = np.array_split(
                ranker_data.select("user_id").unique().collect().to_numpy().flatten(),
                n_splits,
            )
        else:
            batches = [
                ranker_data.select("user_id").unique().collect().to_numpy().flatten()
            ]

        for i in tqdm(range(n_splits)):

            y_pred: np.ndarray = super().predict(
                Pool(
                    data=ranker_data.filter(pl.col("user_id").is_in(batches[i]))
                    .select(SECOND_STAGE_FEATURES)
                    .collect()
                    .to_pandas(),
                    group_id=get_catboost_group_id(
                        ranker_data.filter(pl.col("user_id").is_in(batches[i]))
                    ),
                    cat_features=SECOND_STAGE_CATEGORIAL_FEATURES,
                )
            )

            (
                add_score_and_rank(
                    df=ranker_data.filter(pl.col("user_id").is_in(batches[i])),
                    y_pred_scores=y_pred,
                    name="listwise",
                ).select(FEATURES_FOR_ANALYSIS)
                # Save
                .write_parquet(
                    self.candidates_data_path + f"CB_ranker_predictions_{i}.parquet"
                )
            )

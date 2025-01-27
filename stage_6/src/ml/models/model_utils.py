"""
Module with genereal purpose utils for trainning models
"""
from rectools.dataset import Dataset as RTDataset
from rectools import Columns

import polars as pl


def create_rectools_dataset(models_data: pl.LazyFrame) -> RTDataset:
    """
    Create rectools dataset

    Args:
        models_A Polars LazyFrame containing interaction data with columns 
                    'user_id', 'item_id', 'dt', and 'cum_weight'.

    Returns:
        A rectools Dataset object.
    """
    return RTDataset.construct(
        # Изменим датасетпод стандарт `rectools`
        # Оставим только нужные колонки и переименуем
        interactions_df=models_data.select(
            [
                "user_id",
                "item_id",
                "dt",
                "cum_weight",
            ]
        )
        .rename(
            {
                "user_id": Columns.User,
                "item_id": Columns.Item,
                "dt": Columns.Datetime,
                "cum_weight": Columns.Weight,
            }
        )
        .collect()
        # преобразуем в формат pandas
        .to_pandas(),
    )

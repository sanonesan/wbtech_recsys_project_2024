"""Module with general purpose utils for ranker"""

import numpy as np
import polars as pl


def get_catboost_group_id(df: pl.LazyFrame, group_param: str = "user_id") -> np.ndarray:
    """
    Get CatBoost groups

    Args:
        df (pl.LazyFrame): Pandas DataFrame
        group_param (str): param for grouping
    Returns:
        List: list of groups' ids
    """
    return df.select(group_param).collect().to_numpy().flatten()


def add_score_and_rank(
    df: pl.LazyFrame, y_pred_scores: np.ndarray, name: str
) -> pl.DataFrame:
    """
    Make table from 2-stage model predictions

    Args:
        df (pd.DataFrame): Pandas DataFrame
        y_pred_scores (str): array with model predictions' scores.
        name (str, optional): The name of the column containing predicitons for users

    Returns:
        pd.DataFrame: DataFrame with model's scores and ranks
    """

    df = (
        # Добавляем скор модели второго уровня
        pl.concat(
            [
                df.collect(),
                pl.DataFrame(y_pred_scores, schema=[f"{name}_score"]),
            ],
            how="horizontal",
        )
        # Добавляем ранг модели второго уровня
        .sort(
            by=["user_id", f"{name}_score"],
            descending=[False, True],
        ).with_columns(pl.cum_count("item_id").over("user_id").alias(f"{name}_rank"))
    )

    # Исключаем айтемы, которые не были предсказаны на первом уровне
    mask = (
        (pl.col("cos_rank") < 15)
        | (pl.col("bm25_rank") < 15)
        | (pl.col("lfm_rank") < 15)
        | (pl.col("tfidf_rank") < 15)
    )

    # Добавляем общий скор двухэтапной модели
    eps: float = 0.001
    min_score: float = min(y_pred_scores) - eps
    df = df.with_columns(
        pl.when(mask)
        .then(pl.col(f"{name}_score"))
        .otherwise(min_score)
        .alias(f"{name}_hybrid_score")
    )

    # Добавляем общий ранг двухэтапной модели
    max_rank: int = 101
    df = df.with_columns(
        pl.when(mask)
        .then(pl.col(f"{name}_rank"))
        .otherwise(max_rank)
        .alias(f"{name}_hybrid_rank")
    )

    return df

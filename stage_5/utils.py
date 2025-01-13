"""Module with general purpose utils for 5th stage"""

from typing import Dict, Any

import numpy as np
import pandas as pd


def get_lightgbm_group(df: pd.DataFrame) -> np.ndarray:
    """
    Get LightGBM groups

    Args:
        df (pd.DataFrame): Pandas DataFrame
        group_param (str): param for grouping
    Returns: 
        List: list of groups' ids
    """
    return np.array(
        df[["user_id", "item_id"]].groupby(by=["user_id"]).count()["item_id"]
    )


def get_catboost_group_id(df: pd.DataFrame, group_param: str = "user_id") -> np.ndarray:
    """
    Get CatBoost groups

    Args:
        df (pd.DataFrame): Pandas DataFrame
        group_param (str): param for grouping
    Returns: 
        List: list of groups' ids
    """
    return df["user_id"].tolist()


def add_score_and_rank(
    df: pd.DataFrame, y_pred_scores: np.ndarray, name: str
) -> pd.DataFrame:
    """
    Make table from 2-stage model predictions

    Args:
        df (pd.DataFrame): Pandas DataFrame
        y_pred_scores (str): array with model predictions' scores.
        name (str, optional): The name of the column containing predicitons for users

    Returns:
        pd.DataFrame: DataFrame with model's scores and ranks
    """

    # Добавляем скор модели второго уровня
    df[f"{name}_score"] = y_pred_scores
    # Добавляем ранг модели второго уровня
    df.sort_values(
        by=["user_id", f"{name}_score"],
        ascending=[True, False],
        inplace=True,
    )
    df[f"{name}_rank"] = df.groupby("user_id").cumcount() + 1

    # Исключаем айтемы, которые не были предсказаны на первом уровне
    mask = (
        (df["cos_rank"] < 15)
        | (df["bm25_rank"] < 15)
        | (df["lfm_rank"] < 15)
        | (df["tfidf_rank"] < 15)
    ).to_numpy()

    # Добавляем общий скор двухэтапной модели
    eps: float = 0.001
    min_score: float = min(y_pred_scores) - eps
    df[f"{name}_hybrid_score"] = df[f"{name}_score"] * mask
    df[f"{name}_hybrid_score"].replace(
        0,
        min_score,
        inplace=True,
    )

    # Добавляем общий ранг двухэтапной модели
    df[f"{name}_hybrid_rank"] = df[f"{name}_rank"] * mask
    max_rank: int = 101
    df[f"{name}_hybrid_rank"].replace(
        0,
        max_rank,
        inplace=True,
    )

    return df

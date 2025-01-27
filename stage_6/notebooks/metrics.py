"""
Class for evaluating recommendation systems' metrics
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
import polars as pl


class RecommenderMetrics:
    """
    Class for evaluating the performance of recommendation systems using common metrics.

    This class provides static methods to compute Discounted Cumulative Gain (DCG),
    Normalized Discounted Cumulative Gain (NDCG), Recall, and a combined evaluation
    of NDCG and Recall.
    """

    @staticmethod
    def dcg(scores: np.ndarray) -> float:
        """
        Calculates the Discounted Cumulative Gain (DCG) for a given set of scores.

        DCG is a measure of the usefulness, or gain, of a ranked list of items
        based on their relevance. Items with higher relevance contribute more to the overall score
        and are discounted by their position in the ranked list.

        Args:
            scores (np.ndarray): A 1D numpy array representing relevance scores of items
                in a ranked order.

        Returns:
            float: The Discounted Cumulative Gain (DCG) value.
        """

        return np.sum(
            np.divide(
                np.power(2, scores) - 1,
                np.log2(np.arange(scores.shape[0], dtype=np.float64) + 2),
            ),
            dtype=np.float64,
        )

    @staticmethod
    def ndcg_metric(
        gt_items: np.ndarray, predicted: np.ndarray, k: Optional[int] = None
    ) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG)
        between ground truth items and predicted items.

        NDCG is a metric used to measure the ranking quality of a recommendation system.
        It ranges from 0 to 1, where 1 indicates a perfect ranking
        and 0 indicates a completely irrelevant ranking.

        Args:
            gt_items (np.ndarray): A 1D numpy array representing ground truth item IDs.
            predicted (np.ndarray): A 1D numpy array representing predicted item IDs
                by recommendation system.

        Returns:
            float: The Normalized Discounted Cumulative Gain (NDCG) score,
                ranging from 0 to 1. Returns 0 if ideal_dcg or rank_dcg are 0.
        """
        if k is None:
            k = len(predicted)

        if k < len(predicted):
            predicted = predicted[:k]

        relevance = np.array([1 if x in predicted else 0 for x in gt_items])
        # DCG uses the relevance of the recommended items
        rank_dcg = RecommenderMetrics.dcg(relevance)
        if rank_dcg == 0.0:
            return 0.0

        # IDCG has all relevances to 1 (or the values provided),
        # up to the number of items in the test set that can fit in the list length
        ideal_dcg = RecommenderMetrics.dcg(np.sort(relevance)[::-1][:k])

        if ideal_dcg == 0.0:
            return 0.0

        ndcg_ = rank_dcg / ideal_dcg

        return ndcg_

    @staticmethod
    def recall_metric(
        gt_items: np.ndarray, predicted: np.ndarray, k: Optional[int] = None
    ) -> float:
        """
        Calculates the recall score of the predicted items with respect to ground truth items.

        Recall measures the fraction of relevant items (present in ground truth) that were
        correctly recommended by the system.

        Args:
            gt_items (np.ndarray): A 1D numpy array representing ground truth item IDs.
            predicted (np.ndarray): A 1D numpy array representing predicted item IDs
                by recommendation system.

        Returns:
            float: The recall score.
        """
        if k is None:
            k = len(predicted)

        if k < len(predicted):
            predicted = predicted[:k]

        n_gt = len(gt_items)
        intersection = len(set(gt_items).intersection(set(predicted)))
        return intersection / n_gt

    @staticmethod
    def apk(gt_items, predicted, k: Optional[int] = None) -> float:
        """
        Computes the average precision at k.
        This function computes the average prescision at k between two lists of
        items.

        Args:
            gt_items (np.ndarray): A 1D numpy array representing ground truth item IDs.
            predicted (np.ndarray): A 1D numpy array representing predicted item IDs
                by recommendation system.
            k (optional, int): The maximum number of predicted elements
        Returns:
            score (float): The average precision at k over the input lists
        """
        if k is None:
            k = len(predicted)

        if len(predicted) > k:
            predicted = predicted[:k]

        apk_score = 0.0
        num_hits = 0.0

        for i, item_i in enumerate(predicted):
            # IF CLAUSE is equal to Rel(user, item_i)
            # Where Rel is an indicator function, it equals to
            #   - 1 if the item at rank i is relevant,
            #   - 0 otherwise;
            # Also we do not count if we met this value in predicted earlier
            if (item_i in gt_items) and (item_i not in predicted[:i]):
                num_hits += 1.0
                apk_score += num_hits / (i + 1.0)

        if not len(gt_items):
            return 0.0

        return apk_score / min(len(gt_items), k)

    @staticmethod
    def evaluate_recommender(
        df: pd.DataFrame | pl.DataFrame,
        model_preds_col: str,
        gt_col: str = "item_id",
        k: Optional[int] = 10,
    ) -> Dict[str, float]:
        """
        Evaluates a recommender model based on NDCG and recall metrics.

        This method calculates the average NDCG and Recall scores for each prediction
        in the input DataFrame. The method applies the NDCG and recall metric
        to each row to generate the evaluation metrics.

        Args:
            df (pd.DataFrame): Pandas DataFrame containing the ground truth items
                and the predicted items for each user or session.
                The DataFrame is expected to have a column containing
                ground truth items and column containing model predicted items.
            model_preds_col (str): The name of the column in the DataFrame
                containing model predictions.
            gt_col (str, optional): The name of the column containing ground truth items.
                Defaults to "item_id".

        Returns:
            Dict[str, float]: A dictionary containing the average NDCG and recall score
                over all data in the df.
        """
        metric_values = []

        for _, row in df.iterrows() if isinstance(df, pd.DataFrame) else df.to_pandas().iterrows():
            metric_values.append(
                (
                    RecommenderMetrics.ndcg_metric(
                        row[gt_col], row[model_preds_col], k
                    ),
                    RecommenderMetrics.recall_metric(
                        row[gt_col], row[model_preds_col], k
                    ),
                    RecommenderMetrics.apk(row[gt_col], row[model_preds_col], k),
                )
            )

        return {
            "ndcg@k": np.mean([x[0] for x in metric_values]),
            "recall@k": np.mean([x[1] for x in metric_values]),
            "map@k": np.mean([x[2] for x in metric_values]),
        }

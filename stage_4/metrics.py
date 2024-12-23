"""
Class for evaluating recommendation systems' metrics
"""

from typing import Dict
import numpy as np
import pandas as pd


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
    def ndcg_metric(gt_items: np.ndarray, predicted: np.ndarray) -> float:
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
        at = len(predicted)
        relevance = np.array([1 if x in predicted else 0 for x in gt_items])
        # DCG uses the relevance of the recommended items
        rank_dcg = RecommenderMetrics.dcg(relevance)
        if rank_dcg == 0.0:
            return 0.0

        # IDCG has all relevances to 1 (or the values provided),
        # up to the number of items in the test set that can fit in the list length
        ideal_dcg = RecommenderMetrics.dcg(np.sort(relevance)[::-1][:at])

        if ideal_dcg == 0.0:
            return 0.0

        ndcg_ = rank_dcg / ideal_dcg

        return ndcg_

    @staticmethod
    def recall_metric(gt_items: np.ndarray, predicted: np.ndarray) -> float:
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
        n_gt = len(gt_items)
        intersection = len(set(gt_items).intersection(set(predicted)))
        return intersection / n_gt

    @staticmethod
    def evaluate_recommender(
        df: pd.DataFrame, model_preds_col: str, gt_col: str = "item_id"
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

        for _, row in df.iterrows():
            metric_values.append(
                (
                    RecommenderMetrics.ndcg_metric(row[gt_col], row[model_preds_col]),
                    RecommenderMetrics.recall_metric(row[gt_col], row[model_preds_col]),
                )
            )

        return {
            "ndcg": np.mean([x[0] for x in metric_values]),
            "recall": np.mean([x[1] for x in metric_values]),
        }

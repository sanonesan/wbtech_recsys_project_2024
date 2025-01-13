"""
    Class for handling cold recos
"""

import dill

import numpy as np
import numpy.typing as npt

import pandas as pd

from mab2rec import BanditRecommender

from rectools.dataset import Dataset as RTDataset
from rectools.models import (
    PopularModel,
)

from tqdm.auto import tqdm


class PopularBanditRecommender:
    """
    Class for recommending items with Multi-Armed Bandit
    and popular model
    """

    def __init__(
        self,
        dataset: RTDataset,
        path_bandit_model: str,
        path_popular_model: str,
        top_k=15,
    ):
        """
        Init bandit
        """

        self.dataset = dataset

        with open(path_bandit_model, "rb") as f:
            self.mab_model: BanditRecommender = dill.load(f)
            self.mab_model.top_k = top_k

        with open(path_popular_model, "rb") as f:
            self.popular_model: PopularModel = dill.load(f)

    def __get_arms_for_users(self, user_ids, k=50):
        """
        Get bandit arms
        """
        candidates_pop = self.popular_model.recommend(
            [user_ids[0]],
            self.dataset,
            # выдаем k самых популярных кандидатов на данный момент
            # из которых будет выбирать бандит
            k=k,
            # рекомендуем уже просмотренные товары
            filter_viewed=False,
        )["item_id"].values

        return candidates_pop

    def predict(
        self,
        user_ids: npt.ArrayLike,
        pop_k=500,
        pre_gen_recs=True,
        pre_gen_n=200,
    ):
        """
        Get prediction
        """

        recs = pd.DataFrame()
        bandit_arms = self.__get_arms_for_users(user_ids, k=pop_k)

        # Generate recs from bandit and popular model before hands
        if pre_gen_recs:
            pre_recs = pd.DataFrame()
            cur_recs = pd.DataFrame()

            for n in tqdm(range(pre_gen_n)):
                try:

                    self.mab_model.set_arms(bandit_arms)
                    mab_recs = self.mab_model.recommend(return_scores=True)
                    cur_recs["rec_id"] = [n] * self.mab_model.top_k
                    cur_recs["item_id"] = mab_recs[0]
                    cur_recs["mab_score"] = mab_recs[1]
                    cur_recs["mab_rank"] = [
                        i for i in range(1, self.mab_model.top_k + 1)
                    ]

                    pre_recs = pd.concat([pre_recs, cur_recs])
                except Exception as e:
                    raise e

            cur_recs = pd.DataFrame()
            batches = np.array_split(user_ids, pre_gen_n)

            for i in tqdm(range(len(batches))):
                cur_recs = pre_recs[pre_recs["rec_id"] == i].rename(
                    columns={"rec_id": "user_id"}
                )
                tmp_recs = []
                for user_id in batches[i]:
                    cur_recs["user_id"] = user_id
                    tmp_recs.append(cur_recs)
                recs = pd.concat([recs, pd.concat(tmp_recs)])

            return recs

        cur_recs = pd.DataFrame()

        for user_id in tqdm(user_ids):
            try:

                self.mab_model.set_arms(bandit_arms)
                mab_recs = self.mab_model.recommend(return_scores=True)
                cur_recs["user_id"] = [user_id] * self.mab_model.top_k
                cur_recs["item_id"] = mab_recs[0]
                cur_recs["mab_score"] = mab_recs[1]
                cur_recs["mab_rank"] = [i for i in range(1, self.mab_model.top_k + 1)]

                recs = pd.concat([recs, cur_recs])
            except Exception as e:
                raise e

        return recs

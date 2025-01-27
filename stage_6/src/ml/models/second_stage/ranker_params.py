"""CatBoost Ranker params"""

from catboost.utils import get_gpu_device_count

RANDOM_STATE = 42


CB_EARLY_STOPPING_ROUNDS = 32  # число итераций, в течение которых нет улучшения метрик

CB_PARAMS = {
    "objective": "YetiRank",  # catboost аналог lambdarank, оптимизирующий ndcg и map
    "custom_metric": [
        "NDCG:top=10",
        "NDCG:top=15",
        "NDCG:top=5",
        "NDCG:top=3",
    ],
    "iterations": 750,
    "max_depth": 8,
    "num_leaves": 40,
    "min_child_samples": 124,
    "learning_rate": 0.195,
    "reg_lambda": 1.5,
    "subsample": 0.9,
    "early_stopping_rounds": CB_EARLY_STOPPING_ROUNDS,
    "verbose": CB_EARLY_STOPPING_ROUNDS // 2,  # период вывода метрик
    "random_state": RANDOM_STATE,
    "bootstrap_type": "Bernoulli",  # Change bootstrap_type
    "grow_policy": "Lossguide",
    "task_type": "GPU" if get_gpu_device_count() else "CPU",
}
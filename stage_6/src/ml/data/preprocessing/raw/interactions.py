"""
Module with tools for preprocessing interactions raw data 
"""

import pathlib
from datetime import timedelta

import numpy as np
import numpy.typing as npt
import polars as pl
import dill

from src.logs.console_logger import LOGGER


def __train_test_time_split(
    interactions_path: str,
    data_path: str,
    time_delta: timedelta = timedelta(hours=8),
) -> None:
    """
    Splits interactions data into training and testing sets based on a time delta.

    This function reads interaction data from a Parquet file, selects relevant
    columns, renames them, and then splits the data into training and testing sets
    based on the provided time delta. The training data is then saved to a new
    parquet file named "train_df.parquet"

    Args:
        interactions_path: Path to the input Parquet file containing raw interactions data.
        data_path: Path to the directory where the output "train_df.parquet"
            file will be saved.
        time_delta: Time difference from the maximum date to define the separation
            point between train and test datasets (default is 8 hours).

    Returns:
        None
    """

    interactions_df = (
        # data_load_path + "train_data_10_10_24_10_11_24_final.parquet"
        pl.scan_parquet(interactions_path)
        # Отбираем необходимые колонки
        .select(["wbuser_id", "nm_id", "dt"])
        # Отсортируем по дате
        .sort(by="dt")
        # Для удобства переименуем колонки
        .rename(
            {
                "wbuser_id": "user_id",
                "nm_id": "item_id",
            }
        )
    )

    # Конечная дата
    max_date = interactions_df.select("dt").max().collect().item()

    # Дата начала данных для теста
    train_test_sep = max_date - time_delta

    # Данные для обучения моделей первого
    # и второго уровня (разделение будет потом)
    interactions_df.filter(pl.col("dt") < train_test_sep).collect().write_parquet(
        data_path + "train_df.parquet"
    )

    def format_item_id_test_df(x: pl.List) -> npt.ArrayLike:
        """Formats item IDs based on frequency within a user's viewing history using Polars."""
        if len(x) > 1:
            return x.value_counts().sort("count", descending=True)[""]
        return x

    # Данные для теста
    (
        interactions_df.filter(pl.col("dt") >= train_test_sep)
        .collect()
        .group_by("user_id")
        .agg(pl.col("item_id"))
        .sort(by="user_id")
        .with_columns(pl.col("item_id").map_elements(format_item_id_test_df))
        .write_parquet(data_path + "test_df.parquet")
    )


def __calculate_train_weights(
    data_path: str,
) -> None:
    """
    Calculates interaction weights for user-item pairs in training data.

    This function calculates weights based on the number of interactions
    a user has with a specific item, relative to the total number of
    interactions for that user. The weights are then added to the
    training data and saved as a new parquet file named "train_df_weights.parquet".

    Args:
        data_path: Path to the directory where the input "train_df.parquet"
            file is located and where the output "train_df_weights.parquet"
            file will be saved.

    Returns:
        None
    """
    train_df_weights = (
        pl.scan_parquet(data_path + "train_df.parquet").group_by(["user_id", "item_id"])
        # Посчитаем количество взаимодействий пользователя
        # с каждым конкретным товаром
        .agg(pl.col("item_id").count().alias("ui_inter"))
    )

    train_df_weights = train_df_weights.join(
        train_df_weights.select(["user_id", "ui_inter"])
        .group_by("user_id")
        .agg(pl.col("ui_inter").sum().alias("u_total_inter")),
        on="user_id",
        how="left",
    ).with_columns((pl.col("ui_inter") / pl.col("u_total_inter")).alias("weight"))

    train_df_weights.collect().write_parquet(data_path + "train_df_weights.parquet")


def __calculate_item_rating(
    data_path: str,
):
    """
    Calculates item ratings based on interaction counts and normalizes them
    by the total number of interactions.

    This function reads pre-calculated interaction weights, aggregates interaction
    counts per item, and calculates a normalized item rating by dividing the
    total interactions of an item by the total number of interactions. Finally,
    it saves the item ratings to a new parquet file named "item_rating_df.parquet".

    Args:
        data_path: Path to the directory where the input "train_df_weights.parquet"
            file is located and where the output "item_rating_df.parquet"
            file will be saved.

    Returns:
        None
    """

    # Все взаимодействия с каждым товаром
    item_rating_df = (
        pl.scan_parquet(data_path + "train_df_weights.parquet")
        .select(["item_id", "ui_inter"])
        .group_by("item_id")
        .agg(pl.col("ui_inter").sum().alias("item_count"))
    )

    # Get shape
    n_interactions = (
        pl.scan_parquet(data_path + "train_df.parquet")
        .select("item_id")
        .collect()
        .shape[0]
    )

    # Общий вес\рейтинг товара по всем пользователям
    item_rating_df = item_rating_df.with_columns(
        # нормируем на число взаимодействий
        (pl.col("item_count") / n_interactions).alias("item_rating")
    ).sort(by="item_rating", descending=True)

    item_rating_df.collect().write_parquet(data_path + "item_rating_df.parquet")


def __modify_train(
    data_path: str,
) -> None:
    """
    Modifies the training data by calculating weights, item ratings,
    and adding cumulative weights, and then saves it to disk.

    This function performs the following steps:
        1. Calculates train weights using `__calculate_train_weights`.
        2. Calculates item ratings using `__calculate_item_rating`.
        3. Joins the original training data with the calculated weights.
        4. Removes the intermediate train weights file.
        5. Adds the cumulative weights based on the interaction order to the training data.
        6. Sorts the data by time
        7. Saves the modified training data to a parquet file named "train_df.parquet".

    Args:
        data_path: Path to the directory where the training data and other intermediate
            files are located, and where the final "train_df.parquet"
            file will be saved.

    Returns:
        None
    """
    __calculate_train_weights(data_path)
    __calculate_item_rating(data_path)

    train_df = pl.scan_parquet(data_path + "train_df.parquet")
    train_df_weights = pl.scan_parquet(data_path + "train_df_weights.parquet")

    # Мерджим и сохраняем в parquet
    train_df.join(
        train_df_weights,
        on=["user_id", "item_id"],
        how="left",
    ).collect().write_parquet(data_path + "train_df.parquet")

    # Remove unnessessary file
    pathlib.Path(data_path + "train_df_weights.parquet").unlink()

    (
        pl.scan_parquet(data_path + "train_df.parquet")
        # Подсчет порядковых номеров взаимодействия пользователя с
        # каждым конкретынм товаром
        #
        # Т.е. если взаимодействия с товарами идут в следующем порядке
        # [1, 2, 1, 1, 2, 4]
        #
        # то результат будет следующим:
        #
        # cumcount([1, 2, 1, 1, 2, 4]) + 1 = [1, 1, 2, 3, 2, 1]
        #
        # Можно расчитать быстрее, используя оконную функцию аналогично предыдущему запросу
        # Но у меня не умещается такой вариант в оперативной памяти
        .with_columns(
            (pl.int_range(1, pl.len() + 1))
            .over(["user_id", "item_id"])
            .alias("ui_entry")
        )
        # кумулятивный вес товара на момент просмотра
        .with_columns(
            (pl.col("ui_entry") / pl.col("ui_inter") * pl.col("weight")).alias(
                "cum_weight"
            )
        )
        # сортировать не обязательно, но хорошо будет
        # если отсортировать по времени, т.к. в дальнейшем
        # записи будут делиться по времени и сортировка ускорит процесс:
        # predict блок в в процессоре не будет "спотыкаться"
        .sort(by=["dt"]).collect()
        # сразу сохраним в parquet
        .write_parquet(data_path + "train_df.parquet")
    )


def __train_base_ranker_time_split(
    data_path: str,
    time_delta: timedelta = timedelta(hours=8),
) -> None:
    """
    Splits the training data into datasets for base models and ranker model training based
    on a time delta.

    This function reads the training data, determines a split point based on a time delta
    from the latest interaction, and then saves the data before the split point to
    "base_models_data.parquet" and the data after the split point to "ranker_data.parquet".
    The original training data file ("train_df.parquet") is then deleted.

    Args:
        data_path: Path to the directory containing the "train_df.parquet" file,
            and where the "base_models_data.parquet" and "ranker_data.parquet"
            files will be saved.
        time_delta: The time difference from the maximum date to use as the split point
            between base models and ranker model datasets (default is 8 hours).

    Returns:
        None
    """

    train_df = pl.scan_parquet(data_path + "train_df.parquet")

    # Дата последней интеракиции
    max_data = train_df.select("dt").max().collect().item()

    # Дата разделяющая данные для трейна моделей
    # первого и второго уровней
    base_ranker_sep = max_data - time_delta

    # Данные для обучения моделей первого уровня
    # ranker_data = train_df[(train_df["dt"] >= base_ranker_sep)]
    # Сразу сохраим в бинарник
    train_df.filter(pl.col("dt") >= base_ranker_sep).collect().write_parquet(
        data_path + "ranker_data.parquet"
    )

    # Данные для обучения модели второго уровня
    # base_models_data = train_df[(train_df["dt"] < base_ranker_sep)]
    # Сразу сохраим в бинарник
    train_df.filter(pl.col("dt") < base_ranker_sep).collect().write_parquet(
        data_path + "base_models_data.parquet"
    )

    # Remove unnessessary file
    pathlib.Path(data_path + "train_df.parquet").unlink()


def __split_users_by_data(data_path: str) -> None:
    """
    Extracts and saves unique user IDs from different datasets.

    This function reads user IDs from the base models data, ranker data,
    and test data parquet files, extracts unique IDs, converts them to
    NumPy arrays, reshapes them, and saves the arrays using dill in separate
    files with names "base_users.dill", "ranker_users.dill", and "test_users.dill".

    Args:
        data_path: Path to the directory containing the "base_models_data.parquet",
            "ranker_data.parquet", and "test_df.parquet" files, and where the dill
            files will be saved.

    Returns:
        None
    """

    # Уникальные айдишники пользователей в таблицах

    base_users = (
        pl.scan_parquet(data_path + "base_models_data.parquet")
        .select("user_id")
        .unique()
        .collect()
    ).to_numpy()
    base_users = base_users.reshape(base_users.shape[0])
    # save
    with open(data_path + "base_users.dill", "wb") as f:
        dill.dump(base_users, f)

    ranker_users = (
        pl.scan_parquet(data_path + "ranker_data.parquet")
        .select("user_id")
        .unique()
        .collect()
    ).to_numpy()
    ranker_users = ranker_users.reshape(ranker_users.shape[0])
    # save
    with open(data_path + "ranker_users.dill", "wb") as f:
        dill.dump(ranker_users, f)

    test_users = (
        pl.scan_parquet(data_path + "test_df.parquet")
        .select("user_id")
        .unique()
        .collect()
    ).to_numpy()
    test_users = test_users.reshape(test_users.shape[0])
    # save
    with open(data_path + "test_users.dill", "wb") as f:
        dill.dump(test_users, f)


def __split_users_by_data_intersections(data_path: str) -> None:
    """
    Calculates and saves user set intersections between base, ranker, and test datasets.

    This function loads unique user IDs from "base_users.dill", "ranker_users.dill",
    and "test_users.dill" files. Then it calculates the following user subsets:
        - Users present in both base and ranker datasets (b2r_users).
        - Users only present in the ranker dataset (ranker_only_users).
        - Users present in both base and test datasets (b2t_users).
        - Users present in base or ranker and test datasets (bNr2t_users).
        - Users only present in test dataset (test_only_users).
    The resulting user sets are saved as dill files.

    Args:
        data_path: Path to the directory where the "base_users.dill",
        "ranker_users.dill", and "test_users.dill" files are located,
        and where the new user set dill files will be saved.

    Returns:
        None
    """

    with (
        open(data_path + "base_users.dill", "rb") as f_base,
        open(data_path + "ranker_users.dill", "rb") as f_ranker,
        open(data_path + "test_users.dill", "rb") as f_test,
    ):
        base_users = dill.load(f_base)
        ranker_users = dill.load(f_ranker)
        test_users = dill.load(f_test)

    # Пользователи, которым надо выдавать пресказания для обучения ранкера,
    # т.е. присутствуют и в base_models_data и в ranker_data (base to ranker users)
    b2r_users = np.array(list((set(base_users) & set(ranker_users))))
    # save
    with open(data_path + "b2r_users.dill", "wb") as f:
        dill.dump(b2r_users, f)

    # на оставшихся пользователях ранкер обучаться не будет
    # на них просто не будет скоров
    ranker_only_users = np.array(list(set(ranker_users) - set(base_users)))
    # save
    with open(data_path + "ranker_only_users.dill", "wb") as f:
        dill.dump(ranker_only_users, f)

    # Проверим качество на тестовой выборке
    # Берем только пользователей, которые присутствуют
    # в base и test выборках
    b2t_users = np.array(list(set(test_users) & (set(base_users))))
    with open(data_path + "b2t_users.dill", "wb") as f:
        dill.dump(b2t_users, f)

    # Пользователи из test_df, которым будут выданы
    # таргетирвонные рекомондации
    bNr2t_users = np.array(  # pylint: disable=C0103
        list((set(base_users) | set(ranker_users)) & set(test_users))
    )
    # save
    with open(data_path + "bNr2t_users.dill", "wb") as f:
        dill.dump(bNr2t_users, f)

    # Пользователи, которые присутствуют только в test_df (cold_users)
    test_only_users = np.array(
        list(set(test_users) - (set(base_users) | set(ranker_users)))
    )
    # save
    with open(data_path + "test_only_users.dill", "wb") as f:
        dill.dump(test_only_users, f)


def __split_users_into_groups(data_path: str) -> None:
    """
    Splits users into different groups based on their presence in training, ranking,
    and testing data.

    This function orchestrates the splitting of users into various groups
    by first calling the `__split_users_by_data` function to extract unique user IDs
    from different datasets, and then calling the `__split_users_by_data_intersections`
    function to calculate and save the intersections of these user sets.

    Args:
        data_path: Path to the directory where the data files are located and where
            the output user group files will be saved.

    Returns:
        None
    """
    __split_users_by_data(data_path)
    __split_users_by_data_intersections(data_path)


def preprocess_interactions_pipeline(
    interactions_path: str,
    data_path: str,
    time_delta_train_test_split: timedelta = timedelta(hours=8),
    time_delta_base_ranker_split: timedelta = timedelta(hours=8),
) -> None:
    """
    Preprocesses raw interactions data by performing train/test splits, modifications,
    and user grouping.

    This pipeline performs the following steps:
        1. Splits the interactions data into train and test sets based on a time delta.
        2. Modifies the training data.
        3. Splits the training data further into sets for base models and ranker model
            training based on another time delta.
        4. Splits the users into groups for further processing.

    Args:
        interactions_path: Path to the raw interactions data file.
        data_path: Path to the directory where processed data will be saved.
        time_delta_train_test_split: Time delta used for splitting the data into
            train and test sets (default is 8 hours).
        time_delta_base_ranker_split: Time delta used for splitting the training data
            into base models and ranker model sets (default is 8 hours).

    Raises:
        Exception: If any error occurs during the preprocessing steps, it's logged
                    and re-raised. The type of the error raised will depend on the
                    underlying functions being called.

    Returns:
        None
    """
    try:

        LOGGER.info(msg="Train\\Test time split interactions data: started...")
        __train_test_time_split(
            interactions_path=interactions_path,
            data_path=data_path,
            time_delta=time_delta_train_test_split,
        )
        LOGGER.info(msg="Train\\Test time split interactions data: finished!")

        LOGGER.info(msg="Modifying interactions data: started...")
        __modify_train(data_path=data_path)
        LOGGER.info(msg="Modifying interactions data: finished!")

        LOGGER.info(
            msg="Train base_models \\ ranker_model split interactions data: started..."
        )
        __train_base_ranker_time_split(
            data_path=data_path,
            time_delta=time_delta_base_ranker_split,
        )
        LOGGER.info(
            msg="Train base_models \\ ranker_model split interactions data: finished!"
        )

        LOGGER.info(msg="Splitting users into groups in interactions data: started...")
        __split_users_into_groups(data_path=data_path)
        LOGGER.info(msg="Splitting users into groups in interactions data: finished!")

    except Exception as e:
        LOGGER.error(msg=f"Error occured while preprocessing interactions data: {e}!")
        raise e

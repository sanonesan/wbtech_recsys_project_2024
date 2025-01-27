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
):

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
):
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
):
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
):
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


def __split_users_by_data(data_path):
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


def __split_users_by_data_intersections(data_path):

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
    bNr2t_users = np.array(
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


def __split_users_into_groups(data_path):

    __split_users_by_data(data_path)
    __split_users_by_data_intersections(data_path)


def preprocess_interactions_pipeline(
    interactions_path: str,
    data_path: str,
    time_delta_train_test_split: timedelta = timedelta(hours=8),
    time_delta_base_ranker_split: timedelta = timedelta(hours=8),
):
    try:

        LOGGER.info(msg="Train\Test time split interactions data: started...")
        __train_test_time_split(
            interactions_path=interactions_path,
            data_path=data_path,
            time_delta=time_delta_train_test_split,
        )
        LOGGER.info(msg="Train\Test time split interactions data: finished!")

        LOGGER.info(msg="Modifying interactions data: started...")
        __modify_train(data_path=data_path)
        LOGGER.info(msg="Modifying interactions data: finished!")

        LOGGER.info(
            msg="Train base_models \ ranker_model split interactions data: started..."
        )
        __train_base_ranker_time_split(
            data_path=data_path,
            time_delta=time_delta_base_ranker_split,
        )
        LOGGER.info(
            msg="Train base_models \ ranker_model split interactions data: finished!"
        )

        LOGGER.info(msg="Splitting users into groups in interactions data: started...")
        __split_users_into_groups(data_path=data_path)
        LOGGER.info(msg="Splitting users into groups in interactions data: finished!")

    except Exception as e:
        LOGGER.error(msg=f"Error occured while preprocessing interactions data: {e}!")
        raise e

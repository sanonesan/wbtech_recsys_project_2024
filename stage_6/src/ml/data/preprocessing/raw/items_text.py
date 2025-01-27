from typing import Tuple, List, Optional

import dill

import numpy as np

import polars as pl
import pandas as pd
import pyarrow.parquet as pq

from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder

# -----------------------------
#
# COMMENTED
#
# Reason: quite a long process that should be done once or rarely
# TODO: Logic to check if embedding for item already exists
# And only at this case calculate it
#
# -----------------------------
# import torch
# from transformers import AutoTokenizer, AutoModel


from src.ml.data.feature_utils import ITEM_CATEGORIAL_FEATURES
from src.ml.data.preprocessing.raw.utils.text_utils import (
    which_color,
    chars_formating_dicts,
    chars_dict,
    parse_char_dict,
    get_char_value,
    format_chars,
)

from src.logs.console_logger import LOGGER


def __get_item_colors_table(df: pl.LazyFrame, data_path: str):

    # Save items_colors
    df.select(["nm_id", "colornames"]).rename(
        {
            "nm_id": "item_id",
            "colornames": "color",
        }
    ).with_columns(pl.col("color").map_elements(which_color)).collect().write_parquet(
        data_path + "items_colors.parquet"
    )


def __get_item_characteristics_table(text_data_path: str, data_path: str):
    df_chars = pl.DataFrame()

    for id_batch in pq.read_table(text_data_path).to_batches():
        df: pd.DataFrame = id_batch.to_pandas()[["nm_id", "characteristics"]]

        df = df.rename(columns={"nm_id": "item_id"})

        df["characteristics"] = df["characteristics"].apply(parse_char_dict)

        for char in sorted(list(set(chars_dict.values()))):
            df[char] = df["characteristics"].apply(lambda x: get_char_value(x, char))

        for k, v in chars_formating_dicts.items():
            df[k] = df[k].apply(lambda x: format_chars(x, v))

        df = df.drop(columns="characteristics")

        df_chars = pl.concat([df_chars, pl.from_pandas(df)])

    df_chars.write_parquet(data_path + "df_chars.parquet")


def __extract_item_desctiption_texts(df: pl.LazyFrame, data_path: str):
    (
        df.select(["nm_id", "title", "description"])
        # .filter(pl.col("nm_id").is_in(id_batch))
        .rename({"nm_id": "item_id"})
        .with_columns(
            # Formatting titles
            pl.when(pl.col("title").is_not_null())
            .then(pl.col("title").str.replace(r"\s\s+", " ").str.to_lowercase())
            .otherwise(pl.col("title").map_elements(lambda x: "")),
            # Formatting descriptions
            pl.when(pl.col("description").is_not_null())
            .then(pl.col("description").str.replace(r"\s\s+", " ").str.to_lowercase())
            .otherwise(pl.col("description").map_elements(lambda x: "")),
            # Get len of titles in chars
            pl.col("title").str.len_chars().alias("title_len"),
            # Get len of descriptions in chars
            pl.col("description").str.len_chars().alias("descr_len"),
            # Get len of titles in words
            pl.col("title").str.split(by=" ").list.len().alias("title_word_len"),
            # Get len of descriptions in words
            pl.col("description").str.split(by=" ").list.len().alias("descr_word_len"),
        )
        .collect()
        # save as parquet
        .write_parquet(data_path + "df_descrs.parquet")
    )


# -----------------------------
#
# COMMENTED
#
# Reason: quite a long process that should be done once or rarely
# TODO: Logic to check if embedding for item already exists
# And only at this case calculate it
#
# -----------------------------

# def __get_item_descriptions_embeddings(data_path):

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Возьмем предобченную модель
#     tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
#     model = AutoModel.from_pretrained("cointegrated/rubert-tiny").to(device)

#     # разобъем на батчи для подачи в модель
#     descrs = np.array_split(
#         pl.scan_parquet(data_path + "df_descrs.parquet")
#         .select("description")
#         .collect()
#         .to_numpy()
#         .flatten(),
#         60 * 4,
#     )

#     all_embeddings = []

#     for i in tqdm(range(len(descrs))):

#         sentences = descrs[i].tolist()

#         encoded_input = tokenizer(
#             sentences,
#             padding=True,
#             truncation=True,
#             max_length=124,
#             return_tensors="pt",
#         ).to(device)

#         with torch.no_grad():
#             model_output = model(**encoded_input)

#         embeddings = model_output.pooler_output
#         embeddings = torch.nn.functional.normalize(embeddings).to("cpu")

#         all_embeddings.append(embeddings)

#     # Сохраним в бинарник
#     with open(data_path + "descrs_embs.dill", "wb") as f:
#         dill.dump((torch.cat(all_embeddings)).numpy(), f)


def __lowrank_descriptions_embeddings(data_path, components_to_keep: int = 10):

    # Загрузим данные
    with open(data_path + "descrs_embs.dill", "rb") as f:
        all_embeddings = dill.load(f)

    pca_lowrank = PCA(n_components=components_to_keep)
    all_embeddings = pca_lowrank.fit_transform(all_embeddings)

    return pl.DataFrame(
        all_embeddings,
        schema=[f"txt_emb_pca_{i}" for i in range(components_to_keep)],
    )


def __get_item_desctiptions_table(
    df: pl.LazyFrame, data_path: str, components_to_keep: int = 10
):

    __extract_item_desctiption_texts(df=df, data_path=data_path)

    # -----------------------------
    #
    # COMMENTED
    #
    # Reason: quite a long process that should be done once or rarely
    # TODO: Logic to check if embedding for item already exists
    # And only at this case calculate it
    #
    # -----------------------------
    #
    # __get_item_descriptions_embeddings(data_path=data_path)
    #
    # -----------------------------

    pl.concat(
        [
            pl.scan_parquet(data_path + "df_descrs.parquet").collect(),
            __lowrank_descriptions_embeddings(
                data_path, components_to_keep=components_to_keep
            ),
        ],
        how="horizontal",
    ).write_parquet(data_path + "df_descrs.parquet")


def __get_item_brands_table(df: pl.LazyFrame, data_path: str):
    (
        df.select(["nm_id", "brandname"])
        .rename(
            {
                "nm_id": "item_id",
                "brandname": "brand",
            }
        )
        .with_columns(pl.col("brand").str.to_lowercase())
        .collect()
        .write_parquet(data_path + "df_brands.parquet")
    )


def __encode_cat_features(
    df: pl.LazyFrame, cat_cols: List
) -> Tuple[pl.DataFrame, OrdinalEncoder]:
    """
    Function for enconding categorial features in table
    and replacing values in original table
    """

    default_values_items = {}
    for cat in cat_cols:
        default_values_items[cat] = (
            df.group_by(cat)
            .agg(pl.col(cat).count().alias("count"))
            .sort("count", descending=True)
            .select(cat)
            .first()
            .collect()
            .item()
        )

    # Fit encoder
    encoder = OrdinalEncoder(dtype=np.int64)
    encoder.set_output(transform="polars")
    encoder.fit(
        df.select(cat_cols)
        .collect()
        .with_columns(
            (
                pl.col(col_name).fill_null(value=col_val)
                for col_name, col_val in default_values_items.items()
            )
        )
    )

    # Modify DataFrame
    return (
        pl.concat(
            [
                df.drop(cat_cols).collect(),
                encoder.transform(
                    df.select(cat_cols)
                    .collect()
                    .with_columns(
                        (
                            pl.col(col_name).fill_null(value=col_val)
                            for col_name, col_val in default_values_items.items()
                        )
                    )
                ),
            ],
            how="horizontal",
        ),
        encoder,
    )


def __encode_items_data(
    items_path: str,
    save_items_path: str,
    save_enc_path: Optional[str] = None,
):

    df_items = pl.scan_parquet(items_path)
    df_items, items_can_enc = __encode_cat_features(df_items, ITEM_CATEGORIAL_FEATURES)
    df_items.write_parquet(save_items_path)

    if save_enc_path:
        with open(save_enc_path, "wb") as f:
            dill.dump(items_can_enc, f)


def preprocess_items_text_data_pipline(text_data_path: str, data_path: str):

    try:

        LOGGER.info(msg="Extracting characteristics table: started...")
        __get_item_characteristics_table(text_data_path, data_path)
        LOGGER.info(msg="Extracting characteristics table: finished!")

        df_text = pl.scan_parquet(text_data_path)

        LOGGER.info(msg="Extracting colors table: started...")
        __get_item_colors_table(df_text, data_path)
        LOGGER.info(msg="Extracting colors table: finished!")

        LOGGER.info(msg="Extracting brands table: started...")
        __get_item_brands_table(df_text, data_path)
        LOGGER.info(msg="Extracting brands table: finished!")

        LOGGER.info(msg="Extracting descriptions table: started...")
        __get_item_desctiptions_table(df_text, data_path)
        LOGGER.info(msg="Extracting descriptions table: finished!")

        LOGGER.info(
            msg="Merging characteristics, colors, brands, descriptions tables: started..."
        )
        # Смерджим данные по айтемам
        pl.scan_parquet(data_path + "df_brands.parquet").join(
            other=pl.scan_parquet(data_path + "df_chars.parquet"),
            on="item_id",
        ).join(
            other=pl.scan_parquet(data_path + "items_colors.parquet"),
            on="item_id",
        ).join(
            other=pl.scan_parquet(data_path + "df_descrs.parquet").drop(
                ["title", "description"]
            ),
            on="item_id",
        ).collect().write_parquet(
            data_path + "df_items_text_data.parquet"
        )
        LOGGER.info(
            msg="Merging characteristics, colors, brands, descriptions tables: finished!"
        )

        LOGGER.info(msg="Encoding categorial features in items table: started...")
        __encode_items_data(
            items_path=data_path + "df_items_text_data.parquet",
            save_items_path=data_path + "df_items_text_data.parquet",
            save_enc_path=data_path + "df_items_text_data_encoder.dill",
        )
        LOGGER.info(msg="Encoding items table: finished!")

    except Exception as e:
        LOGGER.error(msg=f"Error occured while preprocessing items text data: {e}!")
        raise e

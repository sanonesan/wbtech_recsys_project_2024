"""
Module with tools for preprocessing text raw data 
"""

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
# FOR FUTURE: Logic to check if embedding for item already exists
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
    """
    Extracts item IDs and colors from a DataFrame, processes the color information,
    and saves to a Parquet file.

    This function takes a Polars LazyFrame, selects the `nm_id` and `colornames` columns,
    renames them to `item_id` and `color`, applies a custom function `which_color` to
    each color string, and saves the processed data to a Parquet file named "items_colors.parquet".

    Args:
        df: A Polars LazyFrame containing item data, including `nm_id` and `colornames` columns.
        data_path: Path to the directory where the "items_colors.parquet" file will be saved.

    Returns:
        None
    """
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
    """
    Extracts, parses, formats, and saves item characteristics from a Parquet file.

    This function reads item characteristics data from a Parquet file in batches,
    parses and formats the characteristics using custom functions, and saves the
    resulting data to a Parquet file named "df_chars.parquet".

    The function performs the following steps:
        1. Reads the input Parquet file using pyarrow in batches.
        2. Converts each batch to a pandas DataFrame, selecting "nm_id" 
            and "characteristics" columns.
        3. Renames "nm_id" to "item_id".
        4. Applies the `parse_char_dict` function to the "characteristics" column.
        5. For each unique characteristic name in `chars_dict.values()` 
            extracts the characteristic value using the `get_char_value` function 
            and creates a new column for it.
        6. Formats the extracted characteristic values by using the `format_chars` function
            and `chars_formating_dicts`.
        7. Drops the original "characteristics" column.
        8. Concatenates the resulting polars DataFrames
        9. Saves the final DataFrame to a Parquet file.

    Args:
        text_data_path: Path to the input Parquet file containing item characteristics data.
        data_path: Path to the directory where the output "df_chars.parquet" file will be saved.

    Returns:
        None
    """

    df_chars = pl.DataFrame()

    for id_batch in pq.read_table(text_data_path).to_batches():
        df: pd.DataFrame = id_batch.to_pandas()[["nm_id", "characteristics"]]

        df = df.rename(columns={"nm_id": "item_id"})

        df["characteristics"] = df["characteristics"].apply(parse_char_dict)

        for char in sorted(list(set(chars_dict.values()))):
            df[char] = df["characteristics"].apply(lambda x: get_char_value(x, char)) # pylint: disable=W0640

        for k, v in chars_formating_dicts.items():
            df[k] = df[k].apply(lambda x: format_chars(x, v)) # pylint: disable=W0640

        df = df.drop(columns="characteristics")

        df_chars = pl.concat([df_chars, pl.from_pandas(df)])

    df_chars.write_parquet(data_path + "df_chars.parquet")


def __extract_item_desctiption_texts(df: pl.LazyFrame, data_path: str):
    """
    Extracts, formats, and saves item title and description texts to a Parquet file.

    This function takes a Polars LazyFrame, selects `nm_id`, `title`, and `description` columns,
    renames `nm_id` to `item_id`, formats title and description text (removes extra spaces
    and lowercases), calculates the length of the titles and descriptions in characters
    and words, and saves the resulting data to a Parquet file named "df_descrs.parquet".

    Args:
        df: A Polars LazyFrame containing item data, including `nm_id`, `title`,
            and `description` columns.
        data_path: Path to the directory where the "df_descrs.parquet" file will be saved.

    Returns:
        None
    """

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
# FOR FUTURE: Logic to check if embedding for item already exists
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


def __lowrank_descriptions_embeddings(data_path: str, components_to_keep: int = 10):
    """
    Reduces the dimensionality of description embeddings using PCA 
    and returns a Polars DataFrame.

    This function loads pre-calculated description embeddings from a dill file,
    applies PCA to reduce the dimensionality to the specified number of components,
    and returns the resulting embeddings as a Polars DataFrame.

    Args:
        data_path: Path to the directory where the "descrs_embs.dill" file is located.
        components_to_keep: The number of principal components to retain (default is 10).

    Returns:
        A Polars DataFrame containing the low-rank description embeddings. The schema of the
        DataFrame consists of columns named:
            "txt_emb_pca_0", "txt_emb_pca_1", ..., "txt_emb_pca_n",
        where n is equal to `components_to_keep`.
    """
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
    """
    Extracts item description texts, reduces their embeddings dimensionality, and saves to Parquet.

    This function performs the following steps:
        1. Extracts and saves item descriptions and their statistics by calling
            `__extract_item_desctiption_texts` function.
        2. Calculates embeddings for item descriptions (commented out, because of
            its long processing time and should be done separately, and it's planned
            to add a logic to check if embeddings already exist, and calculate
            only in case of its absence).
        3. Reduces the dimensionality of the description embeddings using
            `__lowrank_descriptions_embeddings`.
        4. Combines the original item descriptions data and the low-rank
            embeddings into a single DataFrame and saves it to "df_descrs.parquet".

    Args:
        df: A Polars LazyFrame containing item data, including `nm_id`, `title`, and
            `description` columns.
        data_path: Path to the directory where the intermediate and final data files
            will be saved.
        components_to_keep: The number of principal components to retain during
            dimensionality reduction of description embeddings (default is 10).

    Returns:
        None
    """

    __extract_item_desctiption_texts(df=df, data_path=data_path)

    # -----------------------------
    #
    # COMMENTED
    #
    # Reason: quite a long process that should be done once or rarely
    # FOR FUTURE: Logic to check if embedding for item already exists
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
    """
    Extracts item IDs and brands from a DataFrame, formats the brands, 
    and saves to a Parquet file.

    This function takes a Polars LazyFrame, selects `nm_id` and `brandname` columns,
    renames them to `item_id` and `brand` respectively, converts brands to lowercase,
    and saves the resulting data to a Parquet file named "df_brands.parquet".

    Args:
        df: A Polars LazyFrame containing item data, including `nm_id` and `brandname` columns.
        data_path: Path to the directory where the "df_brands.parquet" file will be saved.

    Returns:
        None
    """
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
    Encodes categorical features in a DataFrame using an OrdinalEncoder 
    and returns the transformed DataFrame and the encoder.

    This function does the following:
        1.  Determines the most frequent value for each categorical column.
        2.  Fits an OrdinalEncoder to the categorical columns, filling null values
            with the most frequent values before encoding.
        3.  Transforms the categorical columns, again filling null values with
            the most frequent values, and combines the transformed columns with the
            rest of the DataFrame.
        4.  Returns both the transformed DataFrame and the fitted encoder.

    Args:
        df: A Polars LazyFrame containing the data.
        cat_cols: A list of column names representing the categorical features to encode.

    Returns:
        A tuple containing:
        - The transformed Polars DataFrame with encoded categorical columns.
        - The fitted OrdinalEncoder object.
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
    """
    Encodes categorical features in item data and saves the encoded data and encoder.

    This function performs the following steps:
        1. Reads item data from a Parquet file using `pl.scan_parquet`.
        2. Encodes categorical features in the DataFrame using `__encode_cat_features`.
        3. Saves the encoded DataFrame to a new Parquet file.
        4. If `save_enc_path` is provided, saves the encoder object using `dill`.

    Args:
        items_path: Path to the input Parquet file containing item data.
        save_items_path: Path to where the encoded item data will be saved as a Parquet file.
        save_enc_path: Optional path to save the fitted encoder object as a dill file. If None, 
            encoder object will not be saved.

    Returns:
        None
    """

    df_items = pl.scan_parquet(items_path)
    df_items, items_can_enc = __encode_cat_features(df_items, ITEM_CATEGORIAL_FEATURES)
    df_items.write_parquet(save_items_path)

    if save_enc_path:
        with open(save_enc_path, "wb") as f:
            dill.dump(items_can_enc, f)


def preprocess_items_text_data_pipline(text_data_path: str, data_path: str):
    """
    Preprocesses item text data by extracting, merging, and encoding features, 
    and saving to Parquet files.

    This pipeline performs the following steps:
        1. Extracts item characteristics using `__get_item_characteristics_table`.
        2. Extracts item colors using `__get_item_colors_table`.
        3. Extracts item brands using `__get_item_brands_table`.
        4. Extracts item descriptions and calculates embeddings using 
            `__get_item_desctiptions_table`.
        5. Merges the extracted features (characteristics, colors, brands, and descriptions)
            into a single DataFrame and saves it to "df_items_text_data.parquet".
        6. Encodes the categorical features in the merged DataFrame using
            `__encode_items_data` and saves both the encoded data and the encoder object.

    Args:
        text_data_path: Path to the input Parquet file containing item text data.
        data_path: Path to the directory where intermediate and final processed data
            will be saved.

    Raises:
        Exception: If any error occurs during the preprocessing steps, it's logged
                and re-raised. The type of the error raised will depend on the
                underlying functions being called.

    Returns:
        None
    """

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

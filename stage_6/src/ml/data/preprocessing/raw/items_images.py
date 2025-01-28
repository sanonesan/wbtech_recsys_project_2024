"""
Module with tools for preprocessing images raw data 
"""

import os  # pylint: disable=W0611
import numpy as np  # pylint: disable=W0611
import numpy.typing as npt
import polars as pl
import dill

import tqdm  # pylint: disable=W0611

from sklearn.decomposition import PCA

# -----------------------------
#
# COMMENTED
#
# Reason: quite a long process that should be done once or rarely
# FOR FUTURE: Logic to check if embedding for item already exists or
# data was changed and only at this case calculate it
#
# -----------------------------

# import torch
# from torch.utils.data import Dataset

# from transformers import CLIPProcessor, CLIPModel

# from .utils.image_utils import CustomImageDataset

from src.logs.console_logger import LOGGER


# Размер изображения, к которому будем приводить
IMG_SIZE = 128

# -----------------------------
#
# COMMENTED
#
# Reason: quite a long process that should be done once or rarely
# FOR FUTURE: Logic to check if embedding for item already exists or
# data was changed and only at this case calculate it
#
# -----------------------------

# def __get_items_img_lists(imgs_data_path: str, data_path):

#     # Sorded list with existing items
#     item_id_list = (
#         pl.scan_parquet(data_path + "df_items.parquet")
#         .select("item_id")
#         .sort(by="item_id")
#         .collect()
#         .to_numpy()
#         .flatten()
#         .tolist()
#     )

#     items_without_img = []

#     for i in item_id_list:
#         if not (os.path.exists(imgs_data_path + f"{i}.jpg")):
#             items_without_img.append(i)

#     return item_id_list, item_id_list


# def __get_img_dataset(imgs_data_path: str, data_path, img_size: int = 128):

#     item_id_list, items_without_img = __get_items_img_lists(imgs_data_path, data_path)

#     # Получим отсортированный список лейблов изображений,
#     # для которых присутствуют картинки
#     labels = sorted(list(set(item_id_list) - set(items_without_img)))

#     images_paths = [imgs_data_path + f"{idx}.jpg" for idx in labels]

#     # Инициализируем датасет
#     return CustomImageDataset(
#         images_paths,
#         labels=labels,
#         image_size=img_size,
#     )


# def __get_device():
#     return "cuda" if torch.cuda.is_available() else "cpu"


# def __get_clip_model(device: str = "cpu"):

#     # Load model
#     model: CLIPModel = CLIPModel.from_pretrained(
#         "openai/clip-vit-base-patch32",
#     ).to(device)
#     processor: CLIPProcessor = CLIPProcessor.from_pretrained(
#         "openai/clip-vit-base-patch32",
#     )

#     # Evaluate mod
#     model.eval()

#     return model, processor


# def __extract_clip_embeddings(
#     model,
#     loader,
#     device: str = "cpu",
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     all_embeddings = []
#     all_labels = []

#     with torch.no_grad():  # Отключаем градиенты для ускорения
#         for images, labels in tqdm.tqdm(iter(loader)):
#             # Перемещаем данные на указанное устройство
#             images = images.to(device)

#             # Извлекаем эмбеддинги из модели
#             embeddings = model.get_image_features(images)

#             # Сохраняем результаты
#             all_embeddings.append(embeddings)
#             all_labels.append(labels)

#     # Объединяем все батчи в один тензор
#     all_embeddings = torch.cat(all_embeddings, dim=0).to("cpu")
#     all_labels = torch.cat(all_labels, dim=0)

#     return all_embeddings, all_labels


# def __get_clip_dataloader(
#     processor: CLIPProcessor,
#     img_dataset: Dataset,
#     batch_size: int = 512,
# ):

#     # collate function
#     def imgs_collate_fn(batch):
#         images, labels = zip(*batch)
#         images = [
#             processor(
#                 images=image,
#                 return_tensors="pt",
#             )["pixel_values"]
#             for image in images
#         ]
#         images = torch.cat(images, dim=0)
#         labels = torch.tensor(labels)
#         return images, labels

#     # Get dataloader
#     return torch.utils.data.DataLoader(
#         img_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=imgs_collate_fn,
#     )


# def __get_clip_embeddings(
#     imgs_data_path: str,
#     data_path: str,
#     batch_size: int = 512,
# ):
#     device = __get_device()
#     model, processor = __get_clip_model(device)

#     img_embeds, img_labels = __extract_clip_embeddings(
#         model,
#         __get_clip_dataloader(
#             processor=processor,
#             img_dataset=__get_img_dataset(imgs_data_path, data_path),
#             batch_size=batch_size,
#         ),
#     )

#     # Сохраним в бинарники эмбединги и лейблы катинок
#     with (
#         open(data_path + "img_embeds.dill", "wb") as f_embeds,
#         open(data_path + "img_labels.dill", "wb") as f_labels,
#     ):
#         dill.dump(img_embeds.numpy(), f_embeds)
#         dill.dump(img_labels.numpy(), f_labels)


def __lowrank_img_embeddings(img_embeds: npt.ArrayLike, components_to_keep: int = 10):
    """
    Reduces the dimensionality of image embeddings using Principal Component Analysis (PCA).

    This function applies PCA to reduce the dimensionality of the input image embeddings
    to the specified number of components.  It returns the transformed embeddings.

    Args:
        img_embeds: A NumPy array or similar structure representing the image embeddings.  The
            shape should be (n_samples, n_features), where n_samples is the number of images and
            n_features is the dimensionality of each embedding.
        components_to_keep: The number of principal components to retain (default is 10).

    Returns:
        A NumPy array of shape (n_samples, components_to_keep) containing 
        the low-rank image embeddings.
    """

    pca_lowrank = PCA(n_components=components_to_keep)

    return pca_lowrank.fit_transform(img_embeds)


def __get_images_embedings_table(data_path: str, components_to_keep: int = 10):
    """
    Loads image embeddings and labels, applies dimensionality reduction,
    and saves the result to a Parquet file.

    This function loads image embeddings and corresponding labels from dill files,
    reduces the dimensionality of the embeddings using
    the `__lowrank_img_embeddings` function,
    combines the labels and the reduced embeddings into a Polars DataFrame, and
    saves the DataFrame as a Parquet file named "df_img_embs.parquet".

    Args:
        data_path: Path to the directory where the "img_embeds.dill" and "img_labels.dill"
            files are located, and where the output "df_img_embs.parquet" file will
            be saved.
        components_to_keep: The number of principal components to retain during
            dimensionality reduction (passed to `__lowrank_img_embeddings` default is 10).

    Returns:
        None
    """

    # Загрузим данные
    with (
        open(data_path + "img_embeds.dill", "rb") as f_embeds,
        open(data_path + "img_labels.dill", "rb") as f_labels,
    ):
        pl.concat(
            [
                pl.DataFrame(
                    data=dill.load(f_labels),
                    schema=["item_id"],
                ),
                pl.DataFrame(
                    data=__lowrank_img_embeddings(
                        dill.load(f_embeds), components_to_keep
                    ),
                    schema=[f"img_emb_pca_{i}" for i in range(components_to_keep)],
                ),
            ],
            how="horizontal",
        ).write_parquet(data_path + "df_img_embs.parquet")


def preprocess_items_images_data_pipline(  # pylint: disable=unused-argument
    imgs_data_path: str,
    data_path: str,
    batch_size: int = 512,
):
    """
    Preprocesses item image data by creating image embeddings, merging them with text data,
    and saving to disk.

    This pipeline performs the following steps:
        1. Extracts image embeddings (commented out in the current version).
            This step is skipped because of its long processing time and should
            be done separately. Logic to check for existing embeddings and only
            calculate if they are missing is planned for the future.
        2. Creates a table of image embeddings using `__get_images_embedings_table`.
        3. Merges the image embeddings table with the text data table for items,
            fills missing values with zeros, and saves the result to "df_items.parquet".

    Args:
        imgs_data_path: Path to the directory containing image data (not used in the
            current implementation, but kept for future use).
        data_path: Path to the directory where intermediate and final processed data
            will be saved.
        batch_size: The batch size to use in embeddings calculations (not used in the
            current implementation, but kept for future use).

    Raises:
        Exception: If any error occurs during the preprocessing steps, it's logged
            and re-raised. The type of the error raised will depend on the
            underlying functions being called.
    Returns:
        None
    """

    try:

        LOGGER.info(msg="Extracting images' embeddings: started...")
        # -----------------------------
        #
        # COMMENTED
        #
        # Reason: quite a long process that should be done once or rarely
        # FOR FUTURE: Logic to check if embedding for item already exists or
        # data was changed and only at this case calculate it
        #
        # -----------------------------
        #
        # __get_clip_embeddings(imgs_data_path, data_path, batch_size)
        #
        # -----------------------------

        LOGGER.info(msg="Extracting images' embeddings: finished!")

        LOGGER.info(msg="Get images' embeddings table: started...")
        __get_images_embedings_table(data_path)
        LOGGER.info(msg="Get images' embeddings table: finished!")

        LOGGER.info(
            msg="Merging images' embedding table with images text data: started..."
        )
        # Merge with main items table
        (
            pl.scan_parquet(data_path + "df_items_text_data.parquet")
            .join(
                other=pl.scan_parquet(data_path + "df_img_embs.parquet"),
                on="item_id",
                how="left",
            )
            # Заполняем пропуски товаров, для которых нет изображений
            .fill_nan(0)
            .fill_null(0)
            .collect()
            .write_parquet(data_path + "df_items.parquet")
        )
        LOGGER.info(
            msg="Merging images' embedding table with images text data: finished!"
        )

    except Exception as e:
        LOGGER.error(msg=f"Error occured while preprocessing items images data: {e}!")
        raise e

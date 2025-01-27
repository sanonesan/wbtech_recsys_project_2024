# from PIL import Image

# from torch.utils.data import Dataset


# class CustomImageDataset(Dataset):
#     def __init__(
#         self,
#         file_paths,
#         labels,
#         transform=None,
#         image_size: int = 128,
#     ):
#         self.file_paths = file_paths
#         self.labels = labels
#         self.transform = transform
#         self.image_size = image_size

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, index):
#         image = Image.open(self.file_paths[index]).resize(
#             size=(self.image_size, self.image_size),
#         )
#         label = self.labels[index]

#         if self.transform:
#             image = self.transform(image)

#         return image, label

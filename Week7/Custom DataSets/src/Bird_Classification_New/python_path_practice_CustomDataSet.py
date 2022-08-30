#%%
import os
import pathlib
import torch
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.

    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    # 3. Crearte a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class BirdDataSet(Dataset):
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self,csv_file:str , dataset: str ,targ_dir: str, transform=None) -> None:

        # 3. Create class attributes
        #The file read in is a dataframe, process it base on dataframe.
      self.annotations = pd.read_csv(targ_dir + "/"+ csv_file)

      self.dataframe = self.annotations[self.annotations["data set"] == dataset].reset_index()

      self.paths = [pathlib.Path(folder_path + "/" + p) for p in self.dataframe["filepaths"]]

      # Setup transforms
      self.transform = transform

      # Create classes and class_to_idx attributes
      self.classes, self.class_to_idx = find_classes(targ_dir + "/" + dataset)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.ut
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dat
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.paths[index].parent.stem #<- (c:/a/b/c).parent.stem = (c:/a/b).stem = b
        class_idx = self.class_to_idx[class_name] #<- select the idx from the class name

        return class_name, class_idx
#%%

folder_path = "D:/archive"
csv_file    = "birds.csv"

# annotation = pd.read_csv(folder_path + "/birds.csv")
# dataframe  = annotation[annotation["data set"] == "train"].reset_index()
# paths = [pathlib.Path(folder_path + "/" + p) for p in dataframe["filepaths"]]
#
# frame = dataframe["filepaths"]

# print()
# class_name = paths[0].parent.stem
# print(paths)

train_set = BirdDataSet(csv_file,"train",folder_path)

img = train_set.load_image(0)

img.show()

#<- f ix class_to_idx
# print(train_set.classes)
# print(train_set.class_to_idx)

"""
#Exploring image data
img,label = train_set[0][0],train_set[0][1]

#Classes
class_names = train_set.classes

# Rearrange the order of dimensions
img_permute = img.permute(1, 2, 0)

# Print out different shapes (before and after permute)
#For image to be plotted , permutation must be used.
print(f"Original shape: {img.shape} -> [color_channels, height, width]")
print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

# Plot the image
plt.figure(figsize=(10, 7))
plt.imshow(img.permute(1, 2, 0))
plt.axis("off")
plt.title(class_names[label], fontsize=14)
"""
# %%

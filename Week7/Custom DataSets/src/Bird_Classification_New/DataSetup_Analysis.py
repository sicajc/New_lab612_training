#%%
from pathlib import Path
import torch
from BirdDataSet import BirdsDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#%%
csv_file = "D:/archive/birds.csv"
root_dir = "D:/archive"
transform = None

BATCH_SIZE = 32

#Traversing the folder
def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str or pathlib.Path): target directory

  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# walk_through_dir(root_dir)

#%%
#Visualizing random images
import random
from PIL import Image

image_path = Path("D:/archive/train")

# Set seed
random.seed(42) # <- try changing this and see what happens

# 1. Get all image paths (* means "any combination")
image_path_list = list(image_path.glob("*/*.jpg")) #<-Creating a List of image path for specification

# 2. Get random image path
random_image_path = random.choice(image_path_list) #<-Randomly select image path from the list

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
#Image class is the name of the folder where the image is stored.
#calling parent returns the parent directory path. Calling stem returns the last info of the path you got
image_class = random_image_path.parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")
# img.show()

#%%
#Plotting with matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Turn the image into an array
img_as_array = np.asarray(img)

# Plot the image with matplotlib
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False);

#%%
from torchvision import datasets
# Use ImageFolder to create dataset(s)
train_dir = "D:/archive/train"
test_dir  = "D:/archive/test"
valid_dir = "D:/archive/valid"

# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])


train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

valid_data = datasets.ImageFolder(root=valid_dir,
                                  transform=data_transform)


print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

#%%
#Get class names
class_names = train_data.classes
#<-Returning a dictionary set with index and class name in it
class_dict  = train_data.class_to_idx

print(class_names)
print(class_dict)

#%%
#Exploring image data
img,label = train_data[0][0],train_data[0][1]

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

#%%
# Turn train and test Datasets into DataLoaders
from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=1, # how many samples per batch?
                              num_workers=0, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=1,
                             num_workers=0,
                             shuffle=False) # don't usually need to shuffle testing data

print(train_dataloader, test_dataloader)

#%%
#<- this returns a batch of data from the data loader you specified
#!Remember to set num_workers to 0, otherwise infinite loop might occurs.
img, label = next(iter(train_dataloader))

# Batch size will now be 1, try changing the batch_size parameter above and see what happens
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")



#%%
#Creating custom dataSet
import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List

#To understand how to load the dataset as a CustomDataset the directory structure
#!MUST BE UNDERSTOOD!

# Setup path for target directory
target_directory = train_dir
print(f"Target directory: {target_directory}")

# Get the class names from the target directory
#<- os.scandir will traverse the folder structure
class_names_found = sorted([entry.name for entry in list(os.scandir(train_dir))])
print(f"Class names found: {class_names_found}")

#%%
#  Make function to find classes in target directory
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

classes,class_dictionary = find_classes(train_dir)
print(classes)
print(class_dictionary)

#%%
#Get all image path using directory method
# 1. Subclass torch.utils.data.Dataset
# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):

    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:

        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)

#%%
import pandas as pd
class BirdDataSet(Dataset):
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, dataset: str ,targ_dir: str, transform=None) -> None:

      # 3. Create class attributes
      #The file read in is a dataframe, process it base on dataframe.
      self.annotations = pd.read_csv(f'{targ_dir}/birds.csv')

      self.dataframe = self.annotations[self.annotations["data set"] == dataset].reset_index()

      self.paths = []

      # note: you'd have to update this if you've got .png's or .jpeg's
      for p in self.dataframe["filepaths"]:
          self.paths.append(targ_dir + "/"+ p)

      # Setup transforms
      self.transform = transform

      # Create classes and class_to_idx attributes
      self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
      "Returns one sample of data, data and label (X, y)."
      img = self.load_image(index)
      class_name = self.classes[index]
      class_idx = self.class_to_idx[class_name]

      return classes, class_idx

folder_path = "D:/archive"

classes , class_idx = BirdDataSet("train",folder_path)

print(classes)
print(class_idx)


#%%
"""
valid_set = BirdsDataset("valid",csv_file,root_dir)
test_set  = BirdsDataset("test",csv_file,root_dir)
train_set = BirdsDataset("train",csv_file,root_dir)

print(f"train_set_length:{len(train_set)}")
print(f"test_set_length:{len(test_set)}")
print(f"valid_set_length:{len(valid_set)}")

#Printing an image out from the set
print(train_set.class_to_idx)


valid_dataloader = DataLoader(valid_set,batch_size = BATCH_SIZE,shuffle = True)
test_dataloader  = DataLoader(test_set,batch_size = BATCH_SIZE,shuffle = True)
train_dataloader = DataLoader(train_set,batch_size = BATCH_SIZE,shuffle = True)



#%%


test_size  = 2000
valid_size = 2000

train_size = 58388

BATCH_SIZE = 32
WORKER_NUM = os.cpu_count()




# test_dataloader = DataLoader(dataset=test_set,batch_size = BATCH_SIZE,num_workers = WORKER_NUM,shuffle =True)
# train_dataloader = DataLoader(dataset=train_set,batch_size = BATCH_SIZE,num_workers = WORKER_NUM,shuffle =True)
# valid_dataloader = DataLoader(dataset=valid_set,batch_size = BATCH_SIZE,num_workers = WORKER_NUM,shuffle =True)

print(train_set)

test_img,label = next(iter(train_set))





#%%
"""
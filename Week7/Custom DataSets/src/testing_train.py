#%%
import os
import torch
import data_setup, engine, VGG16, utils,visual
from PIL import Image
import glob
from VGG16 import VGG_model_INFO
from visual import plot_transformed_images,visualization,walk_through_dir
from torchvision import transforms
from pathlib import Path
from torchvision import models
from torchinfo import summary

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("D:/archive")
#walk_through_dir(data_path)

file_path = f"{data_path}birds.csv"
train_dir = f"{data_path}/train"
test_dir  = f"{data_path}/test"

image_path_list = list(data_path.glob("*/*/*.jpg"))

#Loading data
#train_set = BirdsDataset(csv_file=file_path , root_dir=file_path,transform=transforms.ToTensor())
#train_set, test_set = torch.utils.data.random_split(file_path,[])

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.RandomHorizontalFlip(p=0.4),
  transforms.ToTensor()
])

plot_transformed_images(image_path_list, data_transform, n=3)


# %%
#DataLoader class to load the data in
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

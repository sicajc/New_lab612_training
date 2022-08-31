#%%
import torch
from torch.utils.data import DataLoader
from customDataSet import BirdsDataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from torchvision import datasets

#loading data
# dataset = BirdsDataset(csv_file= 'D:\\archive\\birds.csv',root_dir = 'D:\\archive\\',transform=transforms.ToTensor())
#
# print(len(dataset))
# train_set_size = len(dataset) * 0.8
# test_set_size  = len(dataset) - train_set_size
# train_set,test_set = torch.utils.data.random_split(dataset,[train_set_size,test_set_size])

train_dir =  "D://archive//train"
test_dir  = "D://archive//test"
data_transform = transforms.Compose([
            transforms.Resize(size = (64,64)),
            # Flip the images randomly on the horizontal
            transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
            # Turn the image into a torch.Tensor
            transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])



train_set = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_set = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

print(f"Train data:\n{train_set}\nTest data:\n{test_set}")

train_loader = DataLoader(dataset=train_set,batch_size=32,shuffle=True)
test_loader = DataLoader(dataset=test_set,batch_size=32,shuffle=True)

# Visualization
img,label = next(iter(train_loader))

print(f"Image shape: {img.shape}")
print(f"Label shape: {label.shape}")

# %%

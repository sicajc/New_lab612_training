#%%
import os
import cv2
import time
import torch
import data_setup, engine, VGG16, utils,visual
from PIL import Image
import glob
from torch import nn
from VGG16 import VGG_model_INFO
from visual import plot_transformed_images,visualization
from torchvision import transforms
from pathlib import Path
import torchvision
from torchvision import models
from torchinfo import summary
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

#Q: Why are there 400 directories? when it said there are only 200?
#Q: The epoch seems like it just got stuck


# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
image_path = Path("D:/archive/") #<- uses the pathlib for easier access to paths.
train_dir = f"D:/archive/train/"
test_dir = f"D:/archive/test/"
valid_dir = f"D:/archive/valid/"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

### Exploring Dataset
classes = os.listdir(train_dir)
#print(classes)
print("Total Classes: ",len(classes))

#Counting total train, valid & test images
#Why did I get double the size of data?
train_count = 0
valid_count = 0
test_count = 0
for _class in classes:
    #print(_class)
    train_count += len(os.listdir(train_dir +_class))
    valid_count += len(os.listdir(valid_dir +_class))
    test_count += len(os.listdir(test_dir +_class))


print("Total train images: ",train_count)
print("Total valid images: ",valid_count)
print("Total test images: ",test_count)


#Creating list of all images
train_imgs = []
valid_imgs = []
test_imgs = []

for _class in classes:

    for img in os.listdir(train_dir + _class):
        train_imgs.append(train_dir + _class + "/" + img)

    for img in os.listdir(valid_dir + _class):
        valid_imgs.append(valid_dir + _class + "/" + img)

    for img in os.listdir(test_dir + _class):
        test_imgs.append(test_dir + _class + "/" + img)

class_to_int = {classes[i] : i for i in range(len(classes))}

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(), #<-Order matters.
  transforms.Normalize(mean=[0.48235, 0.45882, 0.40784],std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
])

image_path_list = list(image_path.glob("*/*/*.jpg"))
plot_transformed_images(image_path_list, data_transform, n=2)

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader,valid_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    valid_dir = valid_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

print(len(class_names))

#Why this isnt working?
# for images, labels in train_dataloader:
    # fig, ax = plt.subplots(figsize = (10, 10))
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.imshow(make_grid(images, 4).permute(1,2,0))
    # break


#%%
# Create VGG16
#model = VGG16.VGG_net(in_channels=3,out_channels=len(class_names)).to(device)
#VGG_model_INFO()
#AdaotiveAvgPool2d adjusts the output_size of given input as required
#We have to adjust the classifier layer to meet our need

model  = torchvision.models.vgg16(pretrained = True).to(device)

# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.features.parameters():
    param.requires_grad = False

summary(model=model,
        input_size=(BATCH_SIZE, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

#You can know the input size and output size
#After viewing the summary,adjusting it to the output we
model.classifier = nn.Sequential(
    nn.Linear(25088,4096,bias = True),
    nn.ReLU(inplace = True),
    nn.Dropout(p=0.4),
    nn.Linear(4096, 2048, bias = True),
    nn.ReLU(inplace = True),
    nn.Dropout(0.4),
    nn.Linear(2048, 400)
).to(device)

summary(model=model,
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)


# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
# for param in model.features.parameters():
#     param.requires_grad = False

#%%

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 5,gamma = 0.75)

#%%

# Start training with help from engine.py
results = engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             lr_scheduler= lr_scheduler,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

#%%

#Visualization
visualization(model,
              test_dataloader,
              test_dataloader,
              class_names,
              results["epochs"],
              results["train_loss"],
              results["test_loss"],
              results["train_acc"],
              results["test_acc"])

#%%

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="bird_species_classification_using_VGG.pth")

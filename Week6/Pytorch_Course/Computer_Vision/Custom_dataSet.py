#%%
#Different domain contains different domain libraries, to handle different dataSet
#Lots of prebuilt libraries can be used.
#Now we are going to build a food image recognition model
#How do you get your own data sets into pytorch?
#Import pytorch and setting up device agnostic
#Domain libraries
#You would like to look for torch vision domain libraries for your problem s.t. it can be convinient
import torch
from torch import nn

# %%
#Setting device onto gpu if gpu is available
device = "cuda" if torch.cuda.is_available() else "cpu"
# %%
#We will use food 101 dataSets to pratice
#Best practice is first take a small subset of our dataset, then upscale that
#if needed later
#Our dataSet is a subset of food101 dataSets
#Data set starting with 3 classes of food and 1000 images per class
#(750 training, 250 testing)
#Whole point is to speed up how fast we can experiment
#-------------Getting data---------#
import requests
import zipfile
from pathlib import Path

#Setup data to a data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

#if the image folder doesnt exist, download it and prepare it
if image_path.is_dir():
    print(f"{image_path} directory does exist ... skipping download")
else:
    print(f"{image_path} does not exist, creating one...")
    image_path.mkdir(parents=True, exist_ok = True)

# %%
#Downloading pizza,steak,sushi data
with open(data_path/ "pizza_steak_sushi.zip","wb") as f:
    #Specially noted we would need to download the right RAW data.
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downdloading pizza,steak,sushi data...")
    f.write(request.content)

#Unzipping data
with zipfile.ZipFile(data_path/"pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza,steak,sushi data...")
    zip_ref.extractall(image_path)

#%%
#Becoming one with the data(data preparation and data exploration)
#We would spend a lot of time preparing dataSets
import os
def walk_through_dir(dir_path):
    """Walks through dir_path returning its content"""
    for dirpath,dirnames,filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directores and {len(filenames)} images in '{dirpath}'.")


print(walk_through_dir(image_path))
#%%
#Setting up train and testing paths
#We had better follow the generic approach to store your dataSets
#Simply want to get those data out then convert them into tEnsors
train_dir = image_path / "train"
test_dir = image_path / "test"

print(train_dir)
print(test_dir)
# %%
#Visualization of datas
#Get all of the image paths
#Pick a random image path using Python's random.choice()
#Get the image class name using 'pathlib.Path.parent.stem
#Since we are working with images, let's open the image with pillow, opening image with PIL
#We then show the image and print metadata
import random
from PIL import Image

#Set Seed
random.seed(22)

#Get all image paths
#* means anything inside that level of folder
image_path_list = list(image_path.glob("*/*/*.jpg"))

#Picking a random image path
random_image_path = random.choice(image_path_list)
print(random_image_path)



#Parent is the whole folder directory stem is that pizza image
image_class = random_image_path.parent.stem

#Opening image
img = Image.open(random_image_path)

#Print metadata
print(f"Random image path:{random_image_path}")
print(f"Image class:{image_class}")
print(f"Image width:{img.width}")
print(f"Image height:{img.height}")
img.show()

# %%
#Now let's visualize the image with matplotlib.
import numpy as np
import matplotlib.pyplot as plt
#Turning an image into array
img_as_array = np.asarray(img)

#Plot the image with matplotlib
plt.figure(figsize = (10,7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} '-> [height,width,color_channels](HWC)")
plt.axis(False)
# %%
#Now let's scale it up for every random image.
#Turning all the images into pytorch tensors.
#Transforming data into tensors
#1. turn target data into tensors
#2. turn it into 'torch.utils.data.Dataset' and subsequently a 'torch.utils.data.DataLoader'
#DATASET & DATALOADER
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#Writing transform for image
data_transform = transforms.Compose([
     #Resizing images to 64x64
     transforms.Resize(size = (64,64)),
     #Flip the images randomly
     transforms.RandomHorizontalFlip(p  =0.5),
     #Turn the image into a torch.Tensor
     transforms.ToTensor()
     ])

print(data_transform(img).shape)

#%%
#Visualization
#Transfroming and augmentation
#Smaller image size allows fewer computations
def plot_transformed_images(image_paths,transform,n=3,seed = 42):
    """Selecting random images from a path of images and load/transform
    them then plots the original v.s. transformed version.          """

    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k = n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig,ax = plt.subplots(nrows = 1, ncols = 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize:{f.size}")
            ax[0].axis(False)

            #Transform and plot target image
            #Note we have to change the order from
            #(C,H,W) -> (H,W,C) using permute first otherwise we cannot display
            transformed_image = transform(f).permute(1,2,0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nShape:{transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class:{image_path.parent.stem}", fontsize = 16)

plot_transformed_images(image_paths = image_path_list,
                        transform = data_transform,
                        n=3,
                        seed = 42)

# %%
#Loading image classification data using ImageFolder
#Data loader is oing to help us turn our dataset into iterables and we can customize the batch_size
#So our model can see batch_size images at a time so we must cut our data into multiple batches
#Thus we have to turn the dataset into dataLoader
#This turns our datasets into tensors
from torchvision import datasets
train_data = datasets.ImageFolder(root = train_dir,
                                  transform=data_transform,#Transform for data
                                  target_transform = None)#transform for labels/target

test_data = datasets.ImageFolder(root = test_dir,
                                 transform = data_transform)

print(train_data)
print(test_data)

# %%
#Get class name as list
#Useful list needed
class_names = train_data.classes
print(class_names)

#Get class names as dict
class_dict = train_data.class_to_idx
print(class_dict)
# %%
#Checking length of dataset
print(f"Length of dataSet: {len(train_data),len(test_data)}")
print(train_data.samples[0])

# %%
from torch.utils.data import DataLoader
#Our dataLoader would have smaller number of sample known as a batch, turning our read in data into number of batches
import os
BATCH_SIZE = 1
train_dataloader = DataLoader(dataset = train_data,
                              batch_size = 1,
                              num_workers = os.cpu_count(),
                              shuffle = True) #<- It enables parallellism using multiple cpu.

test_dataloader = DataLoader(dataset = test_data,
                             batch_size = 1,
                             num_workers= os.cpu_count(),
                             shuffle=True) #<- It enables parallellism using multiple

print(f"{len(train_dataloader)} , {test_dataloader}\n")

img,label = next(iter(train_dataloader))

#Batch size will be 1, batch size can be changed
print(f"Image shape: {img.shape} -> [batch_size, color_channels,height,width]")
print(f"Label shape: {label.shape}")

#Creating custom dataset, what if dataFolder doesnt exist? Let's create our own dataLoading class
#We would like to write the function which enable us to load data and then turn it into tensor then turning it into number of batches
#Want to be able to load images from files
#Want to be able to get class names from dataset
#want to be able to get classes as dictionary from the dataset
#Creating custom dataset enables us to build dataset on anything
#Not limited to pytorch dataset functions
#Even if we create a dataset, it might not work
#Using a custom dataset often results in us writing more code, leading more issues.

#%%
import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import DataSet
from torchvision import transforms
from typing import Tuple,Dict,List

#Instance of torchvision.datasets.imageFolder()
#Wanna turn datas into list then dictionary, the helper function we want to build.
#Creating helper function to get class names

#1.Get the class names using os.scandir() to traverse a target directory(ideally the directory is in standard image directory format)
#2.Raise error if class names are not found,things wrong with directory structure
#3.Turn class names into a dict and a list and return them

#Setting up target directory
target_directory = train_dir
print(f"Target dir:{target_directory}")

#Getting class name from the target directory
class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory))])
print(class_names_found)

#%%

def find_classes(directory:str) -> Tuple[List[str],Dict[str,int]]: #Input type -> returning outputs
    """Find class folder names in a target directory"""
    #Get class name by scanning target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    #Raising error if class names could not be found
    if not classes:
        raise FileNotFoundError(f"Couldnt find any classes in {directory}... check file structure\n")

    #Create dictionary index labels(computer prefer numbers rather than strings as labels
    class_to_idx = {class_name : i for i,class_name in enumerate(classes)}

    return classes,class_to_idx

find_classes(target_directory)

#Creating custom dataset representing image folder
#We can always subclass data.utils.data classes to extend our functionality.
#To create own custom dataset
#1. subclassing torch.utils.data.dataset
#2. Init our subclass with a target directory(the directory we'd like to get data from)
#as well as transform if we'd like to transform our data
#3.Creating several attributes
#paths - paths of our images
#transform - the transform we'd like to use
#classes - a list of target classes
#class_to_idx - a dict of target class mapped to integer labels
#4. Creating a function to 'load_images()' this function will open the images
#5. Overwrite 'get_item()"" method to return a given sample when passed an index

#Writing a custom dataset class
from torch.utils.data import Dataset

#1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(DataSet):
    """Writing own custom dataset is error prone"""
    #2. Initilize the custom dataset
    def __init__(self,
                 targ_dir:str,
                 transform = None):

        #3 Creating class attributes
        #Get all of the image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))

        #Setting up transform
        self.transform = transform

        #Create Classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index: int) -> Image.Image:
        "Opening an image via a path and returns it"
        image_path = self.paths[index]
        return Image.open(image_path)


    #5 Overwriting __len__()
    def __len__(self) -> int:
        """Returns the total number of samples"""
        return len(self.paths)

    #6. Overwriting __getitem__() method to return a particular sample
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Returns one sample of data, data and label (X,y)."""
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        "Transform if necessary"
        return (self.transform(img), class_idx) if self.transform else (img, class_idx)


#Create Trnasform s.t. we can know what we want to do when creating custom dataset
train_transforms = transforms.Compose([
    transforms.Resize(size = (64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize(size = (64,64)),
    transforms.ToTensor()
])

#Testing out ImageFolderCustom
train_data_custom = ImageFolderCustom(targ_dir = train_dir ,
                                      transform = train_transforms)

test_data_custom = ImageFolderCustom(targ_dir = test_dir ,
                                     transform = test_transforms)


#Remember for Checking for equality between original ImageFolderDataset v.s. ImageFolderCustom
#To see if they return the same outputs.

#-Now creating function to visualize random image from our dataset-#
#Taking in 'Dataset' and a number of other parameters such as class names and how many images for visualization
#To prevent the display getting out of hand, let's cap the number of images to see at 10
#Set random seed for reproducibility
#Get a list of random sample indexes from the target address
#Setting up a matplotlib plot.
#Loop through the random sample images and lpot them with matplotlib
#Making sure the dimensions of our images line up with matplotlib(HWC)

#1.Creating a function to take in a dataset
def display_random_images(dataset : torch.utils.data.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed : int = None):

    #Adjust display if n is too high
    if n>10:
        n=10
        display_shape = False
        print(f"For display,n shouldnt be too large")

    #Setting seed
    if seed:
        random.seed(seed)

    #Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)),k=n)

    #Setting up plot
    plt.figure(figsize = (16,8))

    #Looping through random indexes and plot them with matplotlib
    for i,targ_sample in enumerate(random_samples_idx):
        targ_image,targ_label = dataset[targ_sample][0] , dataset[targ_sample][1]

        #Adjust tensor dimensions for plotting
        targ_image_adjust = targ_image.permute(1,2,0) #Mapping from[color,channels,width] -> [height,width,channel]

        #Plot adjustment
        plt.subplot(1,n,i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"Class: {classes[targ_label]}"
            if display_shape:
                title = f"{title}\nshape:{targ_image_adjust.shape}"

        plt.title(title)

#Display random images from the ImageFolderCustom Dataset
display_random_images(train_data_custom,
                      n = 10,
                      classes = class_names,
                      seed = 42)


#Now we want to turn custom data into dataLoader
#Turning custom loaded images into DataLoader's
from torch.utils.data import DataLoader

train_dataloader_custom = DataLoader(dataset = train_data_custom,
                                     batch_size = BATCH_SIZE,
                                     num_workers = 0,
                                     shuffle = True)

test_dataloader_custom = DataLoader(dataset = test_data_custom,
                                    batch_size = BATCH_SIZE,
                                    num_workers = os.cpu.count(),
                                    shuffle = True)


img_custom,label_custom = next(iter(train_dataloader_custom))

#Data augmentation
#Data transformation can be done using augmentation. There are lots of option available in documentation.
#Making a vareity of transform to image called data augmentation
#Data augmentation is manually adding diversity to training data.
#Viewing the same image but from different perspective.Like applying rotation?mirror?shift?etc....
#Want to increase the diversity for our data.Hopefully results in a more generalizable to unseen data.
#State if the Art models(SOTA)~to help training a more stable data.
#Using new training method, we can actually boost the accuracy to our model~This may lead to another 5 ~ 6 % of boost in accuracy

#Trivialaugment
from torchvision import transforms
train_transform = transforms.Compose([
                        transforms.Resize(size = (224,224)),
                        transforms.TrivisalAugmentWide(num_magnitude_bins = 31),
                        transforms.ToTensor()]
)

test_transform = transforms.Compose([
                                transforms.Resize(size = (224,224)),
                                transforms.ToTensor()
])

print(image_path)

#Get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))
print(image_path_list[:10])

#Plotting random transformed images
plot_transformed_images(
    image_paths = image_path_list,
    transform = train_transform,
    n=3,
    seed = None
)

#Data augmentation allows increase in model training. SOTA

#Building baseline model
#Model 0 : TinyVgg without data augmentation.
#Later we can train it using data augmentation

#Let's replicate the TinYVGG

#Creating transforms and loading data for model0
#Create simple transform
simple_transform = transforms.Compose([
                    transforms.Resize(size = (64,64)),
                    transforms.ToTensor()
])

#Load and transform data
from torchvision import datasets
train_data_simple = datasets.ImageFolder(root = test_dir,transform = simple_transform)
test_data_simple = datasets.ImageFolder(root = test_dir,transform = simple_transform)


#Turning datasets into dataloader
import os
from torch.utils.data import DataLoader

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

#Creating dataLoader
train_dataloader_simple = DataLoader(dataset = train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle = True,
                                     num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(dataset = test_data_simple,
                                    batch_size = BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=NUM_WORKERS)

#Create TinyVGG model class, color image used here instead of gray image
class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from CNN explainer
    """
    def __init__(self,inpurtshape:int,hidden_units:int,output_shape:int) ->None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels = input_shape,
                      out_channels = hidden_units,
                      kernal_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units,
                      out_channels = hidden_units,
                      kernal_size = 3,
                      stride = 1,
                      padding = 1)
            nn.ReLU(),
            nn.MaxPool2d(kernal_size = 2,stride = 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = hidden_units, #Note the hidden units shape here may not be correct
                      out_features = output_shape)
        )

    def forward(self,x):
        #(How to make your GPU goes brrrrrrrrrrrrrrrrrrrrrrr search for it) We want to avoid frequently Memory Access problem
        return self.classifier(self.conv_block_2(self.conv_block_1(x))) #Benifits from operator fusion


torch.manual_seed(42)
model_0 = TinyVGG(input_shape  = 3, #<- colour image so 3 channels
                  hidden_units = 64,
                  output_shape = len(class_names)).to(device)

print(model_0)

#Check the input and output using a dummy forward path
#Try a forward pass on a single image(for model testing)

image_batch, label_batch = next(iter(train_dataloader_simple))
print(image_batch.shape,label_batch.shape)

#trying the model using forward pass
model_0(image_batch.to(device))


#TorchInfo gives you an idea of shapes when the input is going through the model
#First install torchinfo, import if its available
try:
    import torchinfo
except:
    !pip install torchinfo
    import torchinfo

from torchinfo import summary

#Torchinfo is going to do a forwad pass,torchinfo gives you an overview of your inner layer process for better understanding of the model
#Search for torchInfo documentation for more information
summary(model_0,input_size = [1,3,64,64]) #<- remember to pass in the right input size


#----------------------Now perform training-------------------------#
#Again let's replicate the train and test functions for our model
#train_step() takes in a model and dataloader and trains the model on the dataloader
#test_step() takes in a model and dataloader and evaluates the model one the dataloader

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,device = device):
    model.eval()
    test_loss,test_acc = 0,0
    with torch.inference_mode():
        for X, y in dataloader: #We can actually extract batch out using enumeration
            X,y = X.to(device),y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits,y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim = 1)
            test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)


    test_loss = test_loss/len(dataloader)
    test_acc  = test_acc / len(dataloader)
    return test_loss,test_acc


from tqdm.auto import tqdm

#Creating train function that takes in various model parameters
#optimizer + dataloader
def train(model:torch.nn.Module,train_dataloader : torch.data.DataLoader,optimmizer : torch.optimizer.Optimizer,loss_fn : torch.nn.Module = nn.CrossEntropyLoss(),
          epochs : int = 5, device = device):

    #Creating emtpy result dictionary, to keep track of the result for every epoch
    results = {"train_loss" : [], "train_acc":[],"test_loss":[],"test_acc":[]}


    #Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss,train_acc = train_step(model = model,
                                          dataloader = train_dataloader,
                                          loss_fn =loss_fn,
                                          optimizer = optimizer,
                                          device = device)

        test_loss,test_acc = test_step(model= model,
                                       dataloader = test_dataloader,
                                       loss_fn = loss_fn,
                                       device = device)

        #Print out what's happening
        print(f"Epoch:{epoch} | train acc: {train_acc:.4f} | train loss: {train_loss:.4f} | test_loss: {test_loss:.4f}")

        #Updates the results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

#Then eventually leverage our function then train the model
#Setting random seed
torch.manual(42)
torch.cuda.manual_seed(42)


#Set number pf epochs
NUM_EPOCHS = 5

model_0 = TinyVGG(input_shape = 3, hidden_units = 10,output_shape = len(train_data.classes)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model_0.parameters(),lr = 0.001)

#Timer
from timeit import default_timer as timer
start_time = timer()

#Training model_0
model_0_results = train(model = model_0, train_dataloader = train_dataloader_simple,test_dataloader = test_dataloader_simple,optimizer=optimizer , loss_fn = loss_fn, epochs = NUM_EPOCHS)

#eND THE TIMER
end_time = timer()
print(f"Total training time: {end_time - start_time : .3f}")

#Plotting the loss curves of model0, the loss over time.We would like the loss curve to go down over time
#Get the model_0 results keys
model_0_results.keys()

def plot_loss_curves(results:Dict[str,List[float]]):
    """Plotting training curves of result dict"""
    #Get the loss values of the results dictionary(training and test)
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accracy = results["test_acc"]

    epochs = range(len(results["test_loss"]))

    plt.figure(figsize = (15,7))

    plt.subplot(1,2,1)
    plt.plot(epochs,loss,label = "train_loss")
    plt.plot(epochs,test_loss,label = "test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs") #<- can uses steps or epochs of the x-axis
    plt.legend()

    #Plotting the accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs,accuracy,label = "train_accuracy")
    plt.plot(epochs,test_loss,label = "test_loss")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

plot_loss_curves(model_0_results)

#Intepreting loss curves, sometimes overfitting might occurs, or underfitting. These can be evaluated after analyzing the loss curves

#What should an Ideal loss curve look like?
#Loss curves is one of the most helpful way to troubleshoot a data
#In case of Overfitting
#pytorch learning rate scheduler~ Learning rate adjustment over time.
#Using early stopping in pytorch.Save the pattern and model there
#In case of underfitting
#Both can be solved by utilizing transfer learning.
#Solving overfitting might leads to underfitting~
#Use less regularization
#Using Data augmentation can solve the problem of underfitting.Simply googling might solve your problem. www
#Model 1: Dealing underfitting using Data augmentation, increasing the diversity of data by manually adjusting the data
#Create a transform with data Augmentation

from torchvision import transform
#Creating training transform
train_transform_trivial = transforms.Compose([transforms.Resize(size = (64,64)),
                                              transforms.TrivialAugmentWide(num_magnitude_bins = 31)],
                                             transforms.ToTensor())


test_transform_simple = transforms.Compose([transforms.Resize(size = (64,64)),transforms.ToTensor()])

#Turning image folders into dataSets
from torchvision import datasets
train_data_augmented = datasets.ImageFolder(root = train_dir,transform = train_transform_trivial)

test_data_simple = datasets.ImageFolder(root = test_dir,transform = test_transform_simple)

#Turn our datasets into Dataloaders
import os
from torch.utils.data import DataLoader

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

torch.manual_seed(42)
train_dataloader_augmented = DataLoader(dataset = train_data_augmented,batch_size = BATCH_SIZE,shuffle = True,num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(dataset = test_data_simple,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)

#We are actually using the same architecture but with twitched dataSet
model_1 = TinyVGG(input_shape = 3,hidden_units=10, output_shape =len(train_data_augmented.classes)).to(device)

#Setting random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#Set the number of epocs
NUM_EPOCHS = 5

#Setting up loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model_1.parameters(),lr = 0.01)

#Start timer
from timeit import default_timer as timer
start_time = timer()

model_1_results = train(model=model_1)

#Plotting the loss curves of model 1
plot_loss_curves(model_1_results)

#Comparing model results
#Compare the models with each other
#Tools enable this. python tensorboard allows better visualization
#Weights & Bias enables comparison too
#MLFlow - tracking experimental features

#this is the hard coding style of data visualzation
import pandas as pd
model_0_df = pd.DataFrame(model_0_results)
model_1_df = pd.DataFrame(model_1_results)

#Setting up a plot
plt.figure(figsize = (15,10))

#Get number of epochs
epochs = range(len(model_0_df))

#plot train loss
plt.subplot(2,2,1)
plt.plot(epochs,model_0_df["train_loss"],label = "Model 0")
plt.plot(epochs, model_1_df["train_loss"],label = "Model 1")
plt.title("Test Loss")
plt.xlabel("Epochs")
plt.legend()


#plot train accuracy
plt.subplot(2,2,2)
plt.plot(epochs,model_0_df["train_loss"],label = "Model 0")
plt.plot(epochs, model_1_df["train_loss"],label = "Model 1")
plt.title("Test Loss")
plt.xlabel("Epochs")
plt.legend()

#We definitely want the model to works well with Unseen data on testing datasets

#Making prediction on custom image?How to make predictions on custom dataset. The data that is neither in test data nor train data

import requests

custom_image_path = data_path / "04-pizza-dad.jpeg"

if not custom_image_path.is_file():
    with open(custom_image_path):
        requests = requests.get() #<- put the URL you want to download in
        print(f"Download {custom_image_path}")
        f.write(request.content)
else:
    print(f"")

#Loading in a custom image with pytorch
#In tensor form with dataype right i.e. torch.float32
#Of shape 64x64x3
#On the right device
import torchvision

custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))

plt.imshow(custom_image_uint8)
#Knowing the tensor , datatype and shape! of every images

#Loading in custom image and convert it to torch.float32
custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32) #Chaing the type into float32

#Also we have to convert our image into the right size using resize transform
plt.imshow(custom_image.permute(1,2,0))

#Creating transform pipeline to resize image, the size of image will influence the image quality
custom_image_transform = transforms.Compose([transforms.Resize(size = (64,64))])

custom_image_transformed = custom_image_transform(custom_image)

#printing out shapes



#Making a prediction on a custom image iwth a trained pytorch model
#Try making prediction on an image in uint8
with torch.inference_mode():
    model_1(custom_image_uint8.to(device))

#Remember we have to pass our image with the right dimension and adding a batch size by unsqueeze(0)
#Also remember whether our data are running on the right device
#Load the image an turn it into a tensor
#Making sure image was the smae datatype as model(torch.float32)
#Make sure the image was the same shape and the data the mode was trained on, with batch size(1,3,64,64)


#Convert logits -> prediction probabilities
custom_image_pred_probs = torch.softmax(custom_image_pred, dim = 1)

#Converting prediction probabilities -> prediction labels
custom_iamge_pred_labels = torch.argmax(custom_image_pred_probs, dim = 1).cpu() # when plotting we have to convert data back to cpu!

#Putting image prediction together : building a function

#Ideal outcome:
#A function where we pass an image path, the process all of these,chaging data type,changing the shape,and change the machine it runs on.
def pred_and_plot_image(model: torch.nn.Module,
                        image_path:str,
                        class_names: List[str] = None,
                        device = device):
    """Makes a prediction on a target image with a trained model and plots the image and predictions"""
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    #divide image pixel values by 255 to get them between [0,1]
    target_image = target_image / 255.

    #Transform if necessary
    if transform:
        target_image = transform(target_image)

    model.to(device)

    model.eval()
    with torch.inference_mode():
        #Adding an extra dimension to the image(This is the batch dimension)
        target_image = target_image.unsqueeze(0)

        target_image_pred = model(target_image.to(device))

    #Then convert logits -> prediction probabilites
    target_image_pred_probs = torch.softmax(target_image_pred , dim = 1)

    #Convert prediction probabilites -> prediction labels
    target_image_pred_labels = torch.argmax(target_image_pred_probs , dim = 1)

    #plotting image alonstide the prediction and prediction reusult
    #Plotting of matplotlib only works in cpu not gpu
    #Remove batch dimension and rearrange shape to allow plotting
    plt.imshow(target_image.squeeze().permute(1,2,0))

    if class_names:
        title = f""
    else:
        title = f""

    plt.title(title)
    plt.axis(False)

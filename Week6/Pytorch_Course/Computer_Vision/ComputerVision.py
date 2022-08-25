# %%
#Torchvision is the
# torchvision.datasets
# torchvision.models - pretrained vision module.
# torchvision.transforms- vision data manipulation
# torch.utils.data.Dataset base dataset class for pytorch
# torch.utils.data.dataloader - creates a python iterable

from pathlib import Path
import requests
from torch.utils.data import DataLoader
import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

# We will use MNIST dataset for our network.imagent has various infos and dataSets
# using fashionMNIST from torchvision.datasets
# Setting up dataSets
train_data = datasets.FashionMNIST(
    root="data",  # Where to download data to
    train=True,  # Do we want the training dataset?
    download=True,  # do we want to download yes/no?
    # how do we want to transform the data?
    transform=torchvision.transforms.ToTensor(),
    target_transform=None  # how do we want to transform the label
)

test_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

# %%
# Checking out info of dataSet, the input and output shapes
len(train_data)
len(test_data)

image, label = train_data[0]

class_names = train_data.classes
print(f"Class : {class_names}\n")

class_to_idx = train_data.class_to_idx
print(f"Class = {class_to_idx}\n")


# %%
# Visualising the dataSet
print(f"Image shape:{image.shape}")
plt.imshow(image.squeeze())  # <- this allows removing the dimension
plt.title("label")
# %%

plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)

# %%
# Plotting more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    fig.add_subplot(rows, cols, i)
    img, label = train_data[random_idx]
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)

# %%
# Data loading and mini batches
print(train_data)
print(test_data)

# Prepare dataloader, we would like to turn our data into batches
# The reason we train data usually requires millions of data set
# So training in multiple batches leads to better computational efficiency
# It gives our neural network more chances to update its gradients per epoch


# %%
# Seperating dataSets into batch size of 32, 32 images in a batch
# Turning train dataSet into DataLoader

# Setting up batch size hyperparameter
BATCH_SIZE = 32


# Turning dataset into
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

# Checking out what we have created
print(f"DataLoader:{train_dataloader}\n{test_dataloader}")
print(f"Length of train_dataloader:{len(train_dataloader)}\n")
print(f"Length of test_dataloader:{len(test_dataloader)}\n")

# %%
# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape)  # <- (batch size,colour channel,x,y)
print(train_labels_batch.shape)

# %%
# We've finally turn our data into dataloader
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.axis(False)
print(f"Image size:{img.shape}")
print(f"Label:{label},label size:{label.shape}")

# %%
# 3. Model 0 :Build a baseline model
# Best practice to start is building a baseline model
# baseline model: Simple model you will try and improve upon with subsequent models/experiment
# Creating a flatten layer
flatten_model = nn.Flatten()

# Get a single sample
x = train_features_batch[0]
print(x.shape)

# Flattening the sample,trying to condense our 8D model into single 1d vector
output = flatten_model(x)
print(output.shape)
print(output.squeeze().shape)

# %%


class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape),
        )

    def forward(self, x):
        return self.layer_stack(x)


# Setting up model
# Tensor shape is one of the biggest problem in machine learning.
model_0 = FashionMNISTModelV0(
    input_shape=784,  # 28 * 28 Flattened inputs
    hidden_units=10,
    output_shape=len(class_names)
).to("cpu")

print(model_0)

dummy_x = torch.rand([1, 1, 28, 28])
model_0(dummy_x)

# %%
# Torch metric is useful for evaluation
# setting loss,optimizer and evaluation metrics
# Because working with multi-class data,our loss function will be crossEntropyLoss
# Optimizer using SGD
# evaluation metirc working on classification problem
# Different python functions can be called in using requests
import requests
from pathlib import Path

# Download helpfunction from learn pytorch
# Download helper functions from Learn PyTorch repo (if not already downloaded)
#This is used a lot in large machine learning projects.
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    # Note: you need the "raw" GitHub URL for this to work
    request = requests.get(
        "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

#%%
from helper_functions import accuracy_fn

#Setting up loss_function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_0.parameters(),
                            lr = 0.1)

# %%
#Creating function of timing out experiments.
#Machine learning is very experimental
#1. Model's performance(loss and accuracy values etc...)
#2. How fast it runs
#cpu v.s. gpu
from timeit import default_timer as timer
def print_train_time(
    start : float,
    end:float,
    device:torch.device = None):
    """Print difference between start and end time"""
    total_time = end - start
    print(f"Train time on {device} : {total_time : .3f} seconds\n")
    return total_time

start_time = timer()
end_time = timer()
print_train_time(start = start_time, end = end_time, device = "cpu")


#Creating a training loop and training a model on batches of data
#1.Loop through epochs.
#Loop through training batches, perform training steps, calculate the train loss *per batch*
#loop through testing batches, perform testing steps , calculate the test loss per batch
#printing out what;s going on
#time all
#tqdm python- allows us to know how many epochs we have already gone through.
from tqdm.auto import tqdm

#Setting the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

#Set the number of epochs(we will keep this small for faster training time)
epochs = 3

#Create training and test loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    ### Training
    train_loss = 0
    # Add a loop to loop through training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        # 1. Forward pass
        y_pred = model_0(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulatively add up the loss per epoch

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Print out how many samples have been seen
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)} / {len(train_dataloader.dataset)} samples.")

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)

    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # 1. Forward pass
            test_pred = model_0(X)

            # 2. Calculate loss (accumatively)
            test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1)) #Prediction labels.

        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dataloader)

    print(f"\nTrain Loss:{train_loss : .4f} | Test loss:{test_loss : .4f} , Test acc : {test_acc :.4f}")

    #Calculate training time
    train_time_end_on_cpu = timer()
    total_train_time_model_0 = print_train_time(start= train_time_start_on_cpu,
                                                end = train_time_end_on_cpu,
                                                device = str(next(model_0.parameters())))

    print(next(model_0.parameters()).device)
# %%
#Making predictions and get model0 results.
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            #Making data device agnostic
            X,y = X.to(device) , y.to(device)

            # Make predictions with the model
            y_pred = model(X)

            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

# Calculate model 0 results on test dataset
model_0_results = eval_model(model=model_0, data_loader=test_dataloader,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn
)
model_0_results
print(model_0_results)
# %%
#-----------------------------Setting up functions for reusablility and gpu----------#
#Setting up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#Model1 building model with non-linearity
#Creating linear & non-linear data
class FashionMNISTModelV1(nn.Module):
    def __init__(self,
                 input_shape:int,
                 hidden_units:int,
                 output_shape:int):

        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), #flatten inputs into a single vector
            nn.Linear(in_features = input_shape,out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units,out_features = output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)


#Creating instance of model_1
torch.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape = 784,
                              hidden_units = 10,
                              output_shape = len(class_names)).to(device)

next(model_1.parameters()).device


#Creating loss function and optimizer
#Same Workflow~~~~= =
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_1.parameters(),
                            lr = 0.1) #Trying to update model's parameters reducing loss

#Now lets turn our training loop and testing loop into a function for reusability
#Functionizing training and evaluation loops
#training loop - training_step()
#testing loop - test_step()

def train_step(model : torch.nn.Module,
               data_loader : torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device : torch.device = device):

    train_loss, train_acc = 0,0

    #Put model into training mode
    model.train()

    #Add loop to loops through training batches
    for batch,(X,y) in enumerate(data_loader):
        #Put data onto target device
        X,y = X.to(device), y.to(device)

        #Forward pass
        y_pred = model(X)

        #Loss
        loss = loss_fn(y_pred,y)
        train_loss += loss
        #Going from logits -> predictions labels
        train_acc += accuracy_fn(y_true = y, y_pred = y_pred.argmax(dim = 1))

        #Optimizer zero grad
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(data_loader)
    train_acc  /= len(data_loader)
    print(f"Train loss:{train_loss:.5f}| Train acc : {train_acc: .5f}\n")

def test_step(model: torch.nn.Module,
              data_loader : torch.utils.data.DataLoader,
              loss_fn : torch.nn.Module,
              accuracy_fn,
              device:torch.device = device):
    """Perform a testing loop step on model going over data_loader"""

    test_loss,test_acc = 0,0

    #Put the model in eval mode
    model.eval()

    #turning on inference mode context manager
    with torch.inference_mode():
        for X,y in data_loader:
            #Send data to the target device
            X,y = X.to(device), y.to(device)

            #Forward pass
            test_pred = model(X)

            #calculate the loss/acc
            test_loss += loss_fn(test_pred,y)
            test_acc += accuracy_fn(y_true = y,
                                 y_pred = test_pred.argmax(dim=1))

            #Adjusting mertics and printing it out
            test_loss /= len(data_loader)
            test_acc /= len(data_loader)
            print(f"Test loss: {test_loss: .5f} | Train loss: {test_acc: .5f}")

torch.manual_seed(42)
#Measuring time
from timeit import default_timer as timer
train_time_start_on_gpu = timer()

#Set epochs
epochs = 3

#Create optimization and evaluation loop using train_step() and test_step()
for epoch in tqdm(range(epochs)):
    print(f"Epoch:{epoch}\n----------------------")
    train_step(model = model_1,
               data_loader = train_dataloader,
               loss_fn = loss_fn,
               optimizer = optimizer,
               accuracy_fn = accuracy_fn,
               device =device)

    test_step( model = model_1,
               data_loader = test_dataloader,
               loss_fn = loss_fn,
               accuracy_fn = accuracy_fn,
               device =device)

train_time_end_on_gpu = timer()
total_train_time_model_1= print_train_time(start = train_time_start_on_gpu,
                                           end = train_time_end_on_gpu,
                                           device = device)

# %%
#Sometimes depending on data/hardware might find your model trains fasters on CPU than GPU
#Overhead for copying data/model to and from GPU outweighs the compute benefits offered by GPU
#The hardware CPU is better than GPU.
#Making computer go brrrrrrrrrrr...
#Creating results dictionary
model_1_results = eval_model(model = model_1,
                             data_loader = test_dataloader,
                             loss_fn = loss_fn,
                             accuracy_fn=accuracy_fn)
# %%

#%%
import torch
from torch import nn

#Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#CNN explainer has an example of CNN
#Creating Convolutional Neural Network
class FashionMNISTModelV2(nn.Module):
    """Model arch which replicate tiny VGG16 convnet
        LeNET,AlexNet~etc....
    """
    def __init__(self,input_shape:int , hidden_units:int,output_shape:int):
        super().__init__()
        #Block has multiple layers, block comprises of multiple blocks
        self.conv_block_1 = nn.Sequential(
            #A single block we uses, to learn more about the layer see documentation
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size= 3,
                      stride = 1,
                      padding = 1), #Values we can set, hyperparameter exists in our NN
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units,
                      out_channels =hidden_units,
                      kernal_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernal_size = 2)
        )
        self.conv_block_2 = nn.Sequential(
        #A single block we uses, to learn more about the layer see documentation
            nn.Conv2d(in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size= 3,
                stride = 1,
                padding = 1), #Values we can set, hyperparameter exists in our NN
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units,
                out_channels =hidden_units,
                kernal_size = 3,
                stride = 1,
                padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernal_size = 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = hidden_units*0,
                      out_features = output_shape),
        )

    def forward(self,x):
        #We usually replicates other people's architectures then see whether it works on our own problem~
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape = 1,
                              hidden_units= 10,
                              output_shape = len(class_names)).to(device)
#Create a batch of images
images = torch.randn(size = (32,3,64,64))
test_image = images[0]
print(test_image)

#Stepping through nn.Conv2d()
#Creating a single conv2d layer
#Increase of stride leads to smaller output image.
#If you dont know the hyperParmeters copy those that works. www
conv_layer = nn.Conv2d(in_channels=3 , out_channels=10,kernal_size = (3,3) , stride = 1,padding = 0)

conv_output = conv_layer(test_image.unsqueeze())
# %%

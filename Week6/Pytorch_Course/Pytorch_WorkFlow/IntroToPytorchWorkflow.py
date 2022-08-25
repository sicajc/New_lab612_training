#%%
from typing import OrderedDict
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
what_were_covering = {1:"data(prepare and load)",
                      2:"build model",
                      3:"fitting the model to data(training)",
                      4:"making predictions and evaluating a model(inference)",
                      5:"saving and loading a model",
                      6:"putting it all together"}

#to learn more about nn,look at the documentation
from torch import nn #nn contains all building blocks for neural networks


#1. Data preparing
#Anything can be data, excel, image videos,dna,audio song
#Get data into a numerial representation
#Learn from that numeriacal representation
#To showcase this lets make an exmaple of linear regression
#We will use a linear regression formula to make a straight line with known parameters

#Create known parameters
weight = 0.7
bias = 0.3

#Create
start = 0
end = 1
step = 0.02
x = torch.arange(start,end,step).unsqueeze(dim = 1)
y = weight*x + bias

print(f'x = {x[:10]} \n y = {y[:10]}')
#Three datasets
#1. training set 2. validation set and 3. test set
#Validation sometimes 10~20% training set 60~80% final test set 10%~20%

#1. splitting data into training set and test sets
#Create a train/test split
train_split = int(0.8 * len(x))
print(train_split)

x_train,y_train = x[:train_split],y[:train_split]
x_test,y_test = x[train_split:],y[train_split:]

print(f"x_train = \n {x_train}\n")
print(f"y_train = \n {y_train}\n")

#The training set x,y train
len(x_train) , len(y_train) , len(x_test), len(y_test)

#%%
#Data visualization Visualise datas
def plot_predictions(train_data = x_train,
                     train_labels = y_train,
                     test_data = x_test,
                     test_labels = y_test,
                     predictions = None):

    plt.figure(figsize = (10,7))

    #Plotting training data in blue
    plt.scatter(train_data, train_labels, c = "b" , s = 4 , label = "Training data")

    #Plot test data in green
    plt.scatter(test_data,test_labels, c = "g" , s = 4 , label = "Testing data")

    if predictions is not None:
        #plt predictions if they exist
        plt.scatter(test_data,predictions,c = "r", s= 4, label = "Predictions")


    #Show the legend
    plt.legend(prop = {"size":14})

#here blue dots are inputs , we expects green dots as outputs
plot_predictions()

#2. building pytorch modal
#Creating a linear regression model class

#Everything in pytorch inherent nn.Module
#inherent be the subClass
#What this model does
#Start with random values(weight and biases)
#Look at training data and adjust the random values to better represent(or get closer to) the ideal values
#(the weight & bias values used to create the data)
#How does it do so?
#1. Uses gradient descent
#2. BackPropagation

#%%
# Create a Linear Regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
            requires_grad=True, # <- can we update this value with gradient descent?
            dtype=torch.float # <- PyTorch loves float32 by default
        ))

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
            requires_grad=True, # <- can we update this value with gradient descent?
            dtype=torch.float # <- PyTorch loves float32 by default
        ))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)

#Discuss pytorch building classes
#Pytorch model building essentials
#torch.nn - contatins all buildings for computational graphs
#torch.nn.Parameter - what parameters model should follow and learn.
#torch.nn.module - base class for all nn modules
#torch.optim - optimizer for gradient descent
#def forward() All nn.module subclass need this to forward the backpropogation result
#torch.utils.data.dataset
#torch.utils.data.dataloader
#For more documentation uses pytorch cheatSheet

#Now checks the inner of our model
#We can check out our model parameters or what is inside the model

#Creating Random seed
# Set manual seed since nn.Parameter are randomly initialzied
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel()

# Check the nn.Parameter(s) within the nn.Module subclass we created
print(list(model_0.parameters()))
print(OrderedDict(model_0.state_dict()))

#%%
#Then we want the parameters of our module to be close to the parameters
#Weight and bias we want as close as possible.


#Now lets make prediction with our model
#Making prediction using torch.inference_mode()
#Checking our model's predictive power, let's see how well it predicst y_test
#Make predictions with model

# <- Turning off the gradient track, faster function used here
with torch.inference_mode():
    y_preds = model_0(x_test)

print(f'y_predictions : \n {y_preds}')

# %%
#visualise it
plot_predictions(predictions=y_preds)
#Notice is is making rubbish random predictions

# %%
#Training model
#Training is for some *unknown* parameters to some *known* parameters
#From poor representation of data to better representation of data
#We use loss function to analyze how far away we are from the ideal representation
#Cost function is similiar to loss function, just refer them as Loss Function

# Loss function is a function to measure ideal data
# Optimizer: takes into the account the loss of a model and adjusts the model's parameters

# For pytorch we need
# training Loop & testing loop
#We start with L1 MAE mean absoulate error.
#We use MSE loss on regression
#Loss function keep adjust the weight and bias until we reach ideal data
# Setting up a loss function
loss_fn = nn.L1Loss()

#Setting up an otimizer(stohastic gradient descent algorithm)
#learning rate = lr the most important hyper parameter we would like to set.
#Defines how big or small the parameter changes
optimizer = torch.optim.SGD(params = model_0.parameters(),lr = 0.001)

#Now lets build a training Loop and a testing loop in Pytorch
#A couple of things we need in a training loop:
#0. Loop through the data
#1. Forward Pass (This involve data moving through(forward propagation) our model's forward method)
# to make prediction of the data
#2. Loss Calculation(compare forward pass predictions to ground truth labels)
#3. optimizer zero grad
#4. Loss backward - move backwards through the network to calculate the gradients of each
#parameters of our model with respect to the loss
#5. Optimizer


#Start from epoch is one loop through the data
#This is hyperparameter because we;ve set it
epochs = 2000

#Creating empty loss to track values
train_loss_values = []
test_loss_values = []
epoch_count = []


#0. loop through the data
#Later we would embbed this into a function to reuse it over andover again.
#Pass the number of time epochs you want to loop the data
#the learning rate determines the steps stride, if stride too large we overshoot
#if stride too small it tales forever to reach the bottom
for epoch in range(epochs):
    #Setting model to training mode
    model_0.train() #setting all parameter that requires gradients to require gradients

    #1. forward Pass
    y_pred = model_0(x_train) #<- locate forward data in the code.

    #2. calculate the loss
    loss = loss_fn(y_pred,y_train) #<- calculate the loss value
    #print(f'Loss:{loss}\n') #<- notice loss value would gradually approach the ideal data
    #3. optimzer zero grad
    optimizer.zero_grad() #<- We want to reset the gradient, otherwise it would accumulate over time

    #4 perform back propagation
    loss.backward() #<-

    #5. step the optimizer( perform gradient descent)
    #By default how the optimizer changes would accumulate through the loop
    optimizer.step()

    #----------------------------Testing code--------------------------#
    model_0.eval()

    #This turns off different settings in model not needed for evaluation(drop out layer)
    with torch.inference_mode():
        #Turning gradient tracking off, not important, we only care about data
        #with torch,no_grad(): can also be used
        #1. Do the forward path
        test_pred = model_0(x_test)

        #2. Calculate the loss
        test_loss = loss_fn(test_pred , y_test)

    #Printing out what's happening also for visualization
    if epoch%10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy()) #<- must convert tensor back to numpy array to allow plot
        test_loss_values.append(test_loss.detach().numpy())
        print(f'Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test loss: {test_loss}')

    #Print out model state_dict()
    #print(model_0.state_dict())


#Running the epoch loop by loop and visualise it
#Q:Which loss function and optimizer shall I use?
#A:Problem specific, in this example SME is used.But others might varies.
#Notice that the loss is going down
# %%
#Now lets visualise the data to see how close it is to the ideal value
with torch.inference_mode():
    y_preds_new = model_0(x_test)
#Data get updated through back propagation and gradient descent
#Notice that after more training the predictions fits our testing data
plot_predictions(predictions = y_preds)
plot_predictions(predictions = y_preds_new)


#Plotting the train loss curves and test loss curves
plt.figure(figsize = (4,10))
plt.plot(epoch_count, train_loss_values, label = "Train Loss")
plt.plot(epoch_count, test_loss_values, label = "Test loss")
plt.title("Training and test loss curves")
plt.xlabel("Loss")
plt.ylabel("Epochs")
plt.legend()

# %%
#-Eventually we would like to save AND load the model~
#Saving a model in pytorch
#Three main methods for saving and loading models in Pytorch.

# 1.torch.save() - allows saving a pytorch object in python
# 2.torch.load() - load saved pytorch object
# 3.torch.nn.Module.load_state_dict - load model saved state

#Saving pytorch model
from pathlib import Path

#1. Create models Directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True,exist_ok = True)

#2. Creating a model save path
#.PTH is conventional file name for model
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH =  MODEL_PATH/MODEL_NAME

#3. Saving model state dict
print(f'Saving model to: {MODEL_SAVE_PATH}')
torch.save(obj = model_0.state_dict(), f= MODEL_SAVE_PATH)

#Then we try to load the saved model
loaded_model_0 = LinearRegressionModel()

#Load the saved state_dict of model_0(This will update the new instance with updated parameters)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

#Make some predictions with our loaded model
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(x_test)

print(loaded_model_preds)

y_preds == loaded_model_preds


# %%

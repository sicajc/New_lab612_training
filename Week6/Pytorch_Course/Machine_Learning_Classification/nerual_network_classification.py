#%%
import torch
from sklearn.datasets import make_circles

#nn class with pytorch
#classfication is a problem of predicting whether something is one or another

#--------------------------------Make Classfication data set-------------------------#
#Make samples
n_samples = 1000

#Creating circles
X,y = make_circles(n_samples, noise = 0.03 , random_state = 42)
print(f'X:{X} \n y:{y}')

print(f'First 5 samples of X, y:\n {X[:5]}\n{y[:5]}')

#Make dataframe of circle data
import pandas as pd
circles = pd.DataFrame({"X1" : X[:,0],"X2":X[:,1],"label":y})
circles.head(10)

#Visualising the dataframe
#We now
import matplotlib.pyplot as plt
plt.scatter(x = X[:,0], y = X[:,1], c = y , cmap = plt.cm.RdYlBu)
#We would like to classify whether the data belongs to red or blue of the circle

#The data we are working with is often refered to as a toy dataset,
#A bunches of dataSets are available on scikit Learn.
#Check the input and output datas
print(f'{X.shape} \n {y.shape}')

#Viewing first example of features and labels
X_sample = X[0]
y_sample = y[0]

print(f"Values for one sample of X: {X_sample} and same for y:{y_sample}\n")
print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}\n")

#Now turning data into tensors,float 32
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(f'First 5 values of X & y \n {X[:5]} , {y[:5]}')

#Splitting data into training and test sets
#%%
from sklearn.model_selection import train_test_split

#20% of data will be test 80% train
X_train , X_test , y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

print(f'{len(X_train)} , {len(X_test)} , {len(y_train)} , {len(y_test)} , {n_samples}\n')
# %%
#Building a model
#Lets build a model to classify our blue and red dots
#Doing so
#1.Setting up agonistic code for gpu access
#2.Construct a model(subClassing nn.module)
#3.Define a loss function and optimizer
#4.Creating a training and test loop

from torch import nn

#making device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Current device on: {device}\n")

#Now setup device agnostic code, let's create a model that:
#1. Subclasses nn.module almost all models in pytorch subclass nn.module
#2. Create 2 nn.linear() layers that are capable of hadling the shapes of our data
#3.Defines a forward() method that outlines the forward pass(or forward computation)
#4.Instantiate an instance of our model class and send ti to the target device

#A multiLayer neural network is built here.
#Remember to check the sizes of your train data and test data to prevent size error
#Constructing a model that subclasses nn.module
class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        #Create 2 nn.linear() layers capable of handling the shape of our data
        self.layer_1 = nn.Linear(in_features=2,out_features = 5)
        #Hidden layer below with 5 features, and output features same shape as y
        self.layer_2 = nn.Linear(in_features=5,out_features = 1) #<-In_features has to matches up the previous out_feauters

    #Define a forward() method that outlines the forward pass
    def forward(self,x):
        return self.layer_2(self.layer_1(x)) # x -> layer_1 -> layer_2 -> output

#4. Instantiating an instance of our model class and send it to the target device
model_0 = CircleModel().to(device)
print(f"Model:\n{model_0}\n {model_0.state_dict()}")
#%%
#TensorFlow playground allows you to train browser.
#Figma can allows you to draw some easy pics
#We can acutllay replicate the model of our upper code with nn.sequential
#This is an easy way to implement your model
#However when using this cannot extend to further model
model_0 = nn.Sequential(
    nn.Linear(in_features=2,out_features=5),
    nn.Linear(in_features=5,out_features=1)
).to(device)


#Making predictions
#NN sequential is a faster way of implementing models but lack to detail change
print(f"{model_0.state_dict()}")
#%%
untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions : {len(untrained_preds)}), Shape: {untrained_preds.shape}")

#Setting loss function and optimizer
#Which loss function or optimizer should you use?
#Problem specific
#Regression uses MAE or MSE
#For classification uses binary cross entropy or categorical classification
#For optimizers, two common uses SGD and Adam optimizer
#For the loss function uses torch.nn.BCEWithLogitLoss
#Setting up loss function
#loss_fn = nn.BCELoss() uses the output passing through sigmoid.
loss_fn = nn.BCEWithLogitsLoss() #SIGMOID Activation function.
optimizer = torch.optim.SGD(params = model_0.parameters(),lr = 0.1)

#Calculating accuracy - out of 100 examples, what percentage does our model get right?
def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred)) *100
    return acc

#Training model
#1. forward pass
#2. calculate the loss
#3. optimizer zero grad
#4. loss backward
#5. optimizer step step step...

#Viewing first 5 outputs of the forward pass on the test data
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]

print(f'y_logits : \n {y_logits}\n')
print(f'y_test : \n {y_test[:5]}')

#We uses sigmoid to make model logits to turn them into prediction probabilites
#Sigmoid function must be used to activate to get the binary representation
#logits are raw result from the model then we want to convert the result into binary code
#%%
y_pred_probs = torch.sigmoid(y_logits)
print(f'y_pred_probs : \n {y_pred_probs}\n')

y_preds = torch.round(y_pred_probs)

#In full(logits -> pred probs -> pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

#Check for equality
print(torch.eq(y_preds.squeeze(),y_pred_labels.squeeze()))

#%%
#Building a training loop
torch.cuda.manual_seed(42)

#Set the number of epochs
epochs = 100
#push data to target device
X_train,y_train = X_train.to(device),y_train.to(device)
X_test,y_test = X_test.to(device),y_test.to(device)

#Build training and evaluation loop
for epoch in range(epochs):
    #Training
    model_0.train()

    #1. froward pass
    y_logits = model_0(X_train).squeeze()
    y_pred   = torch.round(torch.sigmoid(y_logits)) #turn logits -> pred probs -> probability

    #2 Calculate loss/accuracy
    #Bewayre that BCELoss and BCEWithLogitsLoss takes different inputs
    loss = loss_fn(y_logits,y_train)
    acc = accuracy_fn(y_true = y_train,y_pred= y_pred)

    #3 optimizer zero grad
    optimizer.zero_grad()

    #4 BackPropagation
    loss.backward()

    #5.
    optimizer.step()

    #testing
    model_0.eval()
    with torch.inference_mode():
        #Forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred   = torch.round(torch.sigmoid(test_logits))

        #Caculate test loss/acc
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true = y_test, y_pred = test_pred)

    #Printing out what's happening
    if epoch% 10 == 0:
        print(f'Epoch: {epoch} | Loss : {loss:.5f}, Acc:{acc:.2f}% | Test loss:{test_loss:.5f}, Test acc : {test_acc:.2f}%')

#Notice that the accuracy is 50 50, so you should check the Input dataSet, to see if it is balacned or not?
#The dataSet we use has 500 red and 500 blue, that's why our model is just guessing.
print(circles.label.value_counts())

#%%
#4. Make predictions and evluate the model
#From the metrics, model is not learning anything.
#To inspect it lets make some predictions and make them visual
#Visualise it!!!! Import function called plot_decision_bounary()
#Goku mohandas.
import requests
from pathlib import Path

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

#%%
#Plot decision boundary of the model
#Helper functions help us visualise the data. Can actually be downloaded online to help
#Visualise data so that we know what's going on to improve the model.
plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_0,X_train,y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_0,X_test,y_test)
#%%
#Improving a model.Improving through experimentation.
#1. Adding more layers
#2. Add more hidden layers
#3. fit for longer, increase the epoch
#4. Chaning the activation functions
#5. Change the learning rate(if number too large exploding gradient problem occurs)
#6. Changing the loss function
class CircleModel1(nn.Module):
    #Lets add more hidden layer
    #Increase layer
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features = 2, out_features = 10)
        self.layer_2 = nn.Linear(in_features = 10 , out_features = 10)
        self.layer_3 = nn.Linear(in_features = 10 , out_features = 1)

    def forward(self,x):
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        #These allows speed up.
        return self.layer_3(self.layer_2(self.layer_1(x)))

model_1 = CircleModel1().to(device)
print(model_1)

# %%
#Creating a loss function
loss_fn = nn.BCEWithLogitsLoss()

#Creating an optimizer
optimizer = torch.optim.SGD(params = model_1.parameters(),lr = 0.1)

#Writing a training nad evaluation loop for model_1
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#Train for longer
epochs = 1000

#Putting data on the target device

for epoch in range(epochs):
    ##Train
    model_1.train()
    #1. forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    #2 Calculate the loss/acc
    loss =loss_fn(y_logits,y_train)
    acc = accuracy_fn(y_true = y_train,y_pred = y_pred)

    #3 .
    optimizer.zero_grad()

    #4.
    loss.backward()

    #5.
    optimizer.step()


    #testing
    model_1.eval()
    with torch.inference_mode():
        #forward pass
        test_logits = model_1(X_test).squeeze()
        test_pred   = torch.round(torch.sigmoid(test_logits))

        #calculate the loss
        test_loss = loss_fn(test_logits,y_test)
        test_acc  = accuracy_fn(y_true = y_test,y_pred = test_pred)


    #Printing out what's going on
    if epoch%100 == 0:
        print(f'Epoch: {epoch} | Loss : {loss:.5f}, Acc:{acc:.2f}% | Test loss:{test_loss:.5f}, Test acc : {test_acc:.2f}%')


#Plotting the decision boundary for checking.
plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_1,X_train,y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_1,X_test,y_test)
# %%
# Preaparing data to see if our model can fit a straight line
#One way to troubleshoot problem is to test a smaller problem
#First starting with smaller portion of dataSet. Linear set must work first
#So lets create some data from the linear one
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

#Creating data
X_regression = torch.arange(start,end,step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias

#Create train and test splots
train_split = int(0.8*len(X_regression))
X_train_regression,y_train_regression = X_regression[:train_split],y_regression[:train_split]
X_test_regression,y_test_regression = X_regression[train_split:],y_regression[train_split:]

plot_predictions(train_data = X_train_regression,
                 train_labels = y_train_regression,
                 test_data = X_test_regression,
                 test_labels = y_test_regression)

# %%
#Now we have to adjust model_1 to fit the linear data.
#Beware of the data's number of features and labels

model_2 = nn.Sequential(
    nn.Linear(in_features = 1 , out_features =10),
    nn.Linear(in_features = 10 , out_features =10),
    nn.Linear(in_features = 10 , out_features =1)
).to(device)

print(model_2)

#Loss & optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params = model_2.parameters(),lr = 0.1)

#Train the model
torch.cuda.manual_seed(42)

#Set number of epochs
epochs = 1000

#Put the data on the target device
X_train_regression,y_train_regression = X_train_regression.to(device),y_train_regression.to(device)
X_test_regression,y_test_regression = X_test_regression.to(device),y_test_regression.to(device)

#Training
for epoch in range(epochs):
    y_pred = model_2(X_train_regression)
    loss = loss_fn(y_pred,y_train_regression)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Testing
    model_2.eval()
    with torch.inference_mode():
        test_preds = model_2(X_test_regression)
        test_loss  = loss_fn(test_pred,y_test_regression)

    #Printing out what's going on
    if epoch % 100 == 0:
        print(f"Epoch:{epoch} | Loss : {loss} | Test loss: {test_loss}")


#Turning on evaluation mode
model_2.eval()

#Make predictions(inference)
with torch.inference_mode():
    y_preds = model_2(X_test_regression)


#Plot data with function
plot_predictions(train_data = X_train_regression.cpu(),
                 train_labels = y_train_regression.cpu(),
                 test_data = X_test_regression.cpu(),
                 test_labels = y_test_regression.cpu())
#%%
#Now we have to introduce Non-Linearity to splits to data set for us.
#Otherwise we can only uses straight line for classification.
#Recreating non-linear data
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

n_samples = 1000

X,y = make_circles(n_samples,noise = 0.03, random_state = 42)

plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.RdYlBu)


import torch
from sklearn.model_selection import train_test_split

#Turning data into tensors
X=torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)


#Building model with non-linear activation
#For non-linear data we need non-linear activation
from torch import nn
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features = 2,out_features = 10)
        self.layer_2 = nn.Linear(in_features = 10,out_features = 10)
        self.layer_3 = nn.Linear(in_features = 10 , out_features =1)
        self.relu    = nn.ReLU() #relu is a non-linear activation function
        #sigmoid function also exist, using the sigmoid function

    def forward(self,x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

#Tensor flow playground is a good place to visualise your NN
model_3 = CircleModelV2().to(device)

#Setting loss and optimizer
#Setup for Binary classification
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(),lr = 0.1)

#Random number
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#Put all data on target device
X_train,y_train = X_train.to(device),y_train.to(device)
X_test,y_test = X_test.to(device),y_test.to(device)

#Loop through data
epochs = 3000

for epoch in range(epochs):
    #Training
    model_3.train()

    #forward passing
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    #loss
    loss = loss_fn(y_logits,y_train)
    acc = accuracy_fn(y_true = y_train,y_pred = y_pred)

    #Optimizer zero grad
    optimizer.zero_grad()

    #loss backward
    loss.backward()

    #step the optimizer
    optimizer.step()

    #Test
    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_pred   = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits,y_test)
        test_acc  = accuracy_fn(y_true = y_test,y_pred = test_pred)

    print(f'Epoch: {epoch} | Loss : {loss:.5f}, Acc:{acc:.2f}% | Test loss:{test_loss:.5f}, Test acc : {test_acc:.2f}%')

#%%
#Making predictions
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

print(f'y_pred : {y_preds[:10]}\n y_test : {y_test[:10]}')

#Plotting the decision boundary for checking.
plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_3,X_train,y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_3,X_test,y_test)
#%%

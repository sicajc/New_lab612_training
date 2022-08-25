#%%
#Putting it all together
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

# %%
#!Plot to assist in visualising data
def plot_predictions(train_data ,
                     train_labels ,
                     test_data,
                     test_labels,
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


#!Check if GPU is enable otherwise use cpu, device agonstic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#Creating dataSet using linear regression formula of y = weight * x + bias
weight = 0.2
bias = 0.1

#Create range Values
start = 0
end = 1
step = 0.02

#Create X & Y (features and labels)
#Unsqueeze to prevent dimensionality issuse
X = torch.arange(start , end , step).unsqueeze(dim = 1)
Y = weight * X + bias
print(f'X:\n {X} \n Y:\n {Y}\n')

#%%
#!Splitting Data
train_split = int(0.8*len(X)) #<- meaning taking 80% of dataSet for training
X_train , Y_train = X[:train_split], Y[:train_split]
X_test,Y_test = X[train_split:] , Y[train_split:] #<- 20% of dataSet for testing
print(f' X_TRAIN_LEN = {len(X_train)}\n Y_TRAIN_LEN = {len(Y_train)} \n X_TEST_LEN = {len(X_test)}\n Y_TEST_LEN = {len(Y_test)}\n')


#%%
#!Plotting the Data
plot_predictions(X_train,Y_train,X_test,Y_test)

#%%
#-----------------------------Building A Pytorch Linear Model----------------------#
#!Creating a linear model by subclassing nn.Module
#Note usually we would not need to assign the random weight and bias yourself
#Built in function can handle for you
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        #Use nn.Linear() for creating the model parameters
        #in_features and out_features depends on the dataSet you use
        #For linear, only have 1 input and 1 output so 1 , 1
        #This implement linear transform layer using preexist layer for training
        #To use different layers visit torch.nn for more details
        self.linear_layer = nn.Linear(in_features=1,out_features=1)

    #Overwriting forward method of nnmodule
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

#Setting Random seed
torch.manual_seed(42)
model_1 = LinearRegressionModel()
print(f'model_1 : \n {model_1} \n model1_1_Dict : \n {model_1.state_dict()}')

# %%
#---------------------------------Training a Model-----------------------------#
#Set the model to target device
#!first check the device you're using
next(model_1.parameters()).device

#%%
#Set the model using target device
model_1.to(device)
next(model_1.parameters()).device

print(f'Model states: \n {model_1.state_dict()}\n')

#We need to set up loss function and optimizer
loss_fn = nn.L1Loss()

#Setup our optimizer
optimizer = torch.optim.SGD(params = model_1.parameters(),lr = 0.01)

#Now writing training loop
torch.manual_seed(42)

#!We need to put data on the target device too.
X_train = X_train.to(device)
Y_train = Y_train.to(device)
Y_test = Y_test.to(device)
X_test = X_test.to(device)

#Creating empty list to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

#!Start the training model loop Remember the Song
epochs = 200
for epoch in range(epochs):
    model_1.train()

    #1. foward passing
    Y_pred = model_1(X_train)

    #2 Calculate loss
    loss = loss_fn(Y_pred,Y_train)

    #3 Optimizer
    optimizer.zero_grad()

    #4 Perform backpropagation
    loss.backward()

    #5 Optimizer step
    optimizer.step()

    ##Testing
    model_1.eval()
    with torch.inference_mode():
        TEST_pred = model_1(X_test)

        TEST_loss = loss_fn(TEST_pred,Y_test)

    #Printing out what's going on
    if epoch % 10 == 0:
        # epoch_count.append(epoch)
        # train_loss_values.append(loss.detach().numpy()) #<- must convert tensor back to numpy array to allow plot
        # test_loss_values.append(TEST_loss.detach().numpy())
        print(f'Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test loss: {TEST_loss}')

#%%
#----------------------Making and evaluating predictions----------------------------#
#Turning into evaluation mode
model_1.eval()

#Make predictions on the test data
with torch.inference_mode():
    Y_preds = model_1(X_test)

#Checking out model Predictions
#Note you must convert the prediction reusult back to cpu device to enable plot
plot_predictions(X_train.cpu(),Y_train.cpu(),X_test.cpu(),Y_test.cpu(),predictions = Y_preds.cpu())

#%%
#----------------------------Saving and loading the model-------------------------#
from pathlib import PurePath
# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True,exist_ok = True)

# 2. Create model save path
MODEL_NAME = "python_workflow_model1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Saving the model state dict
print(f"Saving model to :{MODEL_SAVE_PATH}")
torch.save(obj = model_1.state_dict(), f = MODEL_SAVE_PATH)

#!Loading the python model
#Creating a new instance of linear regression model
loaded_model_1 = LinearRegressionModel()

#Load the saved model_1 state_dict
loaded_model_1.state_dict(torch.load(MODEL_SAVE_PATH))

#Converting back to the right device
loaded_model_1.to(device)

print(f'loaded_model_1 :\n {loaded_model_1.state_dict()}\n')

#!Evaluating the loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)

print(f'Are loaded model the same as the original model?\n{loaded_model_1_preds == Y_preds}\n')
print(f'Y_preds :\n {Y_preds}\n')
print(f'loaded_model_1_preds :\n {loaded_model_1_preds}\n')
# %%

#%%
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn
from helper_functions import plot_predictions, plot_decision_boundary
from torchmetrics import *


##Putting it all together with multiclass classification problem
#Creating multiclass data

#Setting hyperPrameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

#1.Creating multi-class data
X_blob,y_blob = make_blobs(n_samples = 1000, n_features =NUM_FEATURES,centers = NUM_CLASSES,cluster_std = 1.5,random_state = RANDOM_SEED)


#2. Turning data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
#!Beware of the data type here we use long instead of usually torch.float
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

#3. Spliting into train and test sets
X_blob_train,X_blob_test,y_blob_train,y_blob_test = train_test_split(X_blob,y_blob,test_size = 0.2,random_state = RANDOM_SEED)

#4.Plot data
plt.figure(figsize=(10,7))
plt.scatter(X_blob[:,0],X_blob[:,1],c = y_blob , cmap = plt.cm.RdYlBu)
# %%
#Creating device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#Building a multi-class classfication
class BlobModel(nn.Module):
    def __init__(self,input_features,out_features,hidden_units = 8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(

            nn.Linear(in_features = input_features,out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units,out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units,out_features = out_features)
        )

    def forward(self,x):
        return self.linear_layer_stack(x)

#Creating an instance of BlobModel
model_4 = BlobModel(input_features = 2,out_features =4,hidden_units = 8).to(device)



print(f"classes of this set {torch.unique(X_blob_train)}")
#%%
#Crating a loss function for multiclass classfication
loss_fn = nn.CrossEntropyLoss()

#Creating optimizer for multi-class classification
optimizer = torch.optim.SGD(params = model_4.parameters(),lr = 0.1)



#%%
#Getting prediction probability for multiclass model(logits)
model_4(X_blob_test.to(device))

#Remember to check data and device is on the same machine.Remember to check with next and .device()
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test.to(device))

y_logits[:10]


def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred)) *100
    return acc


#In binary classification we use sigmoid, in multiclass we uses softmax.
#Converting our model's logit outputs to prediction probs.
#From prediction probabilitires then to prediction labels(the index of the max)
y_pred_probs = torch.softmax(y_logits,dim=1)
y_preds = torch.argmax(y_logits,dim=1)
print(y_preds)
print(y_blob_test)

#Creating training loop and testing loop
#Fit the multi-class model to the data
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#Set number of epochs
epochs = 100

#Put data to the target device
X_blob_train,y_blob_train = X_blob_train.to(device),y_blob_train.to(device)
X_blob_test,y_blob_test = X_blob_test.to(device),y_blob_test.to(device)

#Looping through data
for epoch in range(epochs):
    model_4.train()

    y_logits = model_4(X_blob_train) #Raw output
    y_pred   = torch.softmax(y_logits,dim=1).argmax(dim=1)

    loss = loss_fn(y_logits,y_blob_train)
    acc = accuracy_fn(y_true = y_blob_train,y_pred=y_pred)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    #Test~
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_preds = torch.softmax(test_logits,dim = 1).argmax(dim=1)

        test_loss = loss_fn(test_logits,y_blob_test)
        test_acc  = accuracy_fn(y_true = y_blob_test, y_pred = test_preds)


    print(f'Epoch: {epoch} | Loss : {loss:.5f}, Acc:{acc:.2f}% | Test loss:{test_loss:.5f}, Test acc : {test_acc:.2f}%')

# %%
#LongTensor looking at the example of each function is a good practice for dtype checking.
#Data type error is annoying.

##Further evaluation predictions with multiclass model
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)


#View the first 10 predictions
print(y_logits[:10])

#Fomr logis -> predictions probability
y_pred_probs = torch.softmax(y_logits,dim=1)
print(y_pred_probs[:10])

#Going from pred probs to pred labels
#Evaluate model visually.
y_preds = torch.argmax(y_pred_probs,dim = 1)
plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
plt.title("train")
plot_decision_boundary(model_4,X_blob_train,y_blob_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_4,X_blob_test,y_blob_test)

# %%
#In this dataset even if you removes ReLU, the code would still works, since the dataSet can
#be sepearated with stragiht lines.
#A few more classification mertics for model evaluation.
#Accuracy - out of 100 samples, how many does out model get right?(used when balance class)
#Precision -
#Recall -
#f1-score
#Confusion matrix
#classification report
#python has inbuilt functions for these, you need to know when you use these.
#When to use precision/recall?
#Precision/recall tradeoff, they actually also suggest you which scenario you should use in documentation
#torchMetrics.
torchmetric_accuracy = Accuracy().to(device)

#Calculating accuracy
torchmetric_accuracy(y_preds,y_blob_test)

# %%

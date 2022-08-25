#%%
from pathlib import Path
import requests
from torch.utils.data import DataLoader
import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from ComputerVision import train_step,test_step,print_train_time
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

#Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

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
                      kernel_size = 3,
                      stride = 1,
                      padding = 1), #Values we can set, hyperparameter exists in our NN
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units,
                      out_channels =hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
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
                kernel_size = 3,
                stride = 1,
                padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        #Note you might encounter error after flatten, the input of layer
        #Must be same as the last stage output!We can check it by viewing shape
        #Each time we passing through a certain layer to prevent the error
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = hidden_units*7*7, #<- how to calculate it ahead of time?
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
conv_layer = nn.Conv2d(in_channels=3 , out_channels=10,kernel_size = (3,3) , stride = 1,padding = 0)

conv_output = conv_layer(test_image.unsqueeze(dim = 0)) #S.t. we removes size error, increasing dimension
# %%
#Stepping through 'nn.MaxPool2d()'
#If having size error, test and see the dimension
#Convolution compress the image data into a single identification vector.

#Creating a sample maxPooling layer
max_pool_layer = nn.MaxPool2d(kernel_size = 2)

#Passing data through just conv_layer
test_image_through_conv = conv_layer(test_image.unsqueeze(dim = 0))
print(f"Shape after going through conv_layer():{test_image_through_conv}\n")

#Passing data through max pool
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)

#Creat a random tensor with similir dimension to our image
random_tensor = torch.randn(size = (1,1,2,2))

#Passing random tensor through max pool
max_pool_tensor = max_pool_layer(random_tensor)
print(f"\nMax Pool tensor:\n {max_pool_tensor}")
print(f"\nMax pool shape\n{max_pool_tensor.shape}")
# %%
#Replication of model then test with your own data.
#Note you must beware which shape you use for your model,
#We need to make sure the output and input size of each layer.
#Training CNN
#Setting up loss & optimizer
from helper_functions import accuracy_fn
from timeit import default_timer as timer
torch.cuda.manual_seed(42)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_2.parameters(),lr = 0.1)

train_time_start_model_2= timer()

epochs = 10
for epoch in tqdm(range(epochs)):
    print(f"Epoch:{epoch}\n----------")
    train_step(model_2,
               data_loader = train_dataloader,
               loss_fn = loss_fn,
               optimizer = optimizer,
               accuracy_fn = accuracy_fn,
               device = device)

    test_step(model = model_2,
              data_loader = test_dataloader,
              loss_fn = loss_fn,
              accuracy_fn = accuracy_fn,
              device = device)

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start = train_time_start_model_2,
                                            end = train_time_end_model_2,
                                            device = device)
# %%
#Getting result of model_2
from ComputerVision import eval_model,model_0_results,model_1_results

model_2_results = eval_model(
    model = model_2,
    data_loader = test_dataloader,
    loss_fn = loss_fn,
    accuracy_fn = accuracy_fn,
    device = device
)

print(model_2_results)

#Comparing model results and training time
import pandas as pd
compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])

#Visualization of our model results
compare_results.set_index("model_name")["model_acc"].plot(kind = "barh")
plt.xlabel("accuracy(%)")
plt.ylabel("model")

#Making predictions of our built CNN using random picked samples
#Make and evaluate random predictions with best model
def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            #Prepare the sample(add a batch dimension and pass to target device)
            sample = torch.unsqueeze(sample,dim = 0).to(device)

            #Forward pass(outputing logits)
            pred_logit = model(sample)

            #Get prediction probability
            pred_prob = torch.softmax(pred_logit.squeeze(),dim = 0)

            #Getting pred_prob off the GPU for further calculation
            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)


import random
random.seed(42)
test_samples =[]
test_labels = []
for sample,label in random.sample(list(test_data), k =9):
    test_samples.append(sample)
    test_labels.append(label)


print(test_samples[0].shape)

# %%
#Visualization of data with truth amd predicted labels
#Models also tell your data as well.

#Making confusion matrix for model evaluation for multiclass classification
#A fantastic way for module evaluation!
# Make predictions with our trained model on the test dataset
# make a confusion matrix
# Plotting the confusion matrix
from tqdm.auto import tqdm

#Make predictions with trained model
y_preds = []
model_2.eval()
with torch.inference_mode():
    for X,y in tqdm(test_dataloader,desc = "Making predictions"):
        #Send data and targests onto target device
        X,y = X.to(device),y.to(device)
        #Doing forward pass
        y_logit = model_2(X)
        #Turn predictions from logits -> prediction probs -> prediction labels
        y_pred = torch.softmax(y_logit.squeeze(),dim = 0).argmax(dim = 1)
        #Put prediction on CPU for evaluation
        y_preds.append(y_pred.cpu())

#Concatenate list of predictions into tensor
#print(y_preds)
#We use confusion matrix to better assess our model's data or model , whats wrong with them then make correction.
#Test test test~
#Visualization visualization visualization
y_pred_tensor = torch.cat(y_preds) #<- turning list of tensors into a single tensor

try:
    import torchmetrics,mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")
    assert int(mlxtend.__version__.split(".")[1] >= 19, "mlxtend version must be 0.19.0 greaters")
except:
    !pip install -q torchmetrics -U mlxtend
    import torchmetrics,mlxtend
    print(f"mlxtend version:{mlxtend.__version__}")


#Eventually we would like to save our model and load up our model.
#torch metrics enables data visualizations
#Evaluation of loaded model after saving is important

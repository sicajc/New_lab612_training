#%%
import pandas as pd
import torch

#%%
print(torch.__version__)

#%%

#scalar
#everything in pytorch work in torch.tensor

scalar = torch.tensor(7)
scalar
#%%

#To look at dimension of tensor
scalar.ndim
#%%
#Giving back python integer
scalar.item()
#Vector
#Magnitude and direction
vector = torch.tensor([7,7])
vector

#%%
#Matrix
MATRIX = torch.tensor([[7,8],[3,4]])

MATRIX.ndim;
MATRIX[1];
MATRIX.shape

#%%
#1 bracket means 1 dimension, more brackets means higher dimension
TENSOR = torch.tensor([[[3,4,5],[1,2,3],[4,5,6]]])
TENSOR.shape

# %%

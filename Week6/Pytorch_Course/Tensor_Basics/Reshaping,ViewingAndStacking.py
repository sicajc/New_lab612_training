#%%
import torch
#Reshaping - reshape an input tensor to a defined shape
#View - return a view of an input tensor of certain shape but keep same memory as the original tensor
#Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
#Squeeze - removes all 1 dimesions from a tensor
#Unqueeze - add a 1 dimension to a target tensor
#Permute - Return a view of the input with dimension permuted in a certain way

#Creating a tensor
x = torch.arange(1.,10.)
print(x , x.shape)

#%%
#Add extra dimension
x_reshaped = x.reshape(3,3) #The number of elements must remain the same
print(x_reshaped, x_reshaped.shape)
print(x.reshape(9,1)) #Row then column
#%%
#Changing view, view share the same memory as the original input
#View works similar like reshape
z = x.view(1,9)
print(z)
z[:0] = 5
print(z)
#%%
#Stacking tensors on top
x_stacked = torch.stack([x,x,x,x],dim = 0)
print(x_stacked)
print(torch.stack([x,x,x,x],dim = 1))

# %%

#%%
import torch
#We usally start with a random numbers -> look at data ->
#update random numbers->look at data -> updata the random numbers


#Random tensors are important because the way many neural networks learns is that they start with tensors full of random numbers
#and then adjust those random numbers to better represent the data
#Creating a random tensor in pytorch of size (3,4)

random_tensor = torch.rand(3,4)
random_tensor
#%%

random_tensor.ndim

#%%
#Creating a random tensor with similar shape to an image tensor
#We splits the image into colour channels of RGB
random_img_size_tensor = torch.rand(size=(224,224,3))

#(torch.Size([224, 224, 3]), 3)
random_img_size_tensor.shape, random_img_size_tensor.ndim
#%%
#Tensors of all zeroes and all ones
zeros = torch.zeros(size=(3,4))
zeros
#Zeroing out the tensor

# %%
#Creating tensor of all ones with default data type float32
ones = torch.ones(size=(3,4))
print(ones)
ones.dtype

# %%
#Creating a range of tensors and tensors-like
#Using torch.range
torch.arange(0,10)

# %%
#Step can also be defined, look at the documentation
#Adding steps
one_to_ten = torch.arange(start = 1, end = 11 , step = 3)
print(one_to_ten)
# %%
#Creating tensors like
#Replicate the shape of another tensor and replacing it all with zeroes
tens_zeros = torch.zeros_like(input = one_to_ten)
print(tens_zeros)
# %%

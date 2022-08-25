#%%
import torch


x = torch.rand(3,2,1)
print(x)
print(x.shape)
#Squeeze removes all single dimensions from a target tensor
#Remove extra dimensions from x
print(f'\nNew tensor:\n{x.squeeze()}\n New Shape:{x.squeeze().shape}')


#%%
#Unsqueeze - adds a single dimension to target tensor at specific dimension
x_unsqueezed = x.unsqueeze(dim = 0)
print(f'\nUsqueezed:\n{x_unsqueezed}')

#%%
#torch.permute - rearranges the dimensions of a target tensor
#Commonly used in image processing, we want to switch the channels
#These allows us to fix value of a certain tensor to allow processing.

x_original = torch.rand(size = (4,5,3)) #[Height,width,colour channels]
x_permuted = x_original.permute(2,0,1)
print(f'x_original:\n{x_original} with shape {x_original.shape}\n x_permuted:\n{x_permuted} with shape {x_permuted.shape}\n')
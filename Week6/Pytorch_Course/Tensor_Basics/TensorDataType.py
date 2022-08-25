#%%
import torch

#Tensors datatypes is one of the 3 big erros you'll run into with Pytorch and deep learning
#1. tensors not right datatype
#2. tensors not right shape
#3. tensors not on the right device

#float32 tensor
#datatpye of tensor are float16 or float32!
#Data type is extremely important, it is related to precision in computer
#Device: We use cuda, cuda and gpu is different, different device leads to error
#Requires_grad: Whether to track gradients
float_32_tensor = torch.tensor([3.0,6.0,9.0],dtype = torch.float32,device = None,requires_grad=False)
print(float_32_tensor)

# %%1
#Changing dtype
float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor)

print(float_16_tensor * float_32_tensor)
# %%
#Some operations might lead to data type error some might not
#If you lead to error in dataype we have to change the datatype manually
#We need to check the tensor's dtype and then convert it into the right dtype
#1. tensor not right dtype uses tensor.dtype
#2. tensor not right shape uses tensor.shape
#3. tensor not on right device uses tensor.device

#Find out details about tensors
some_tensor = torch.rand(3,4)
some_tensor

#%%
#Finding out details about some tensors which helps resolving these common error
print(some_tensor)
print(f"DATATYPE of tensor:{some_tensor.dtype}")
print(f"Shape of tensor:{some_tensor.shape}")
print(f"Device of tensor:{some_tensor.device}")

#%%

#%%
from time import time
import torch

###Manupulating tensors
#Operations
#Addition, subtraction ,mult ,division,matrix,mult

tensor = torch.tensor([1,2,3])
print(tensor + 100)

#%%
#Subtract all entry
print(tensor - 10)

#mult
print(tensor*10)

#%%
##Matrix multiplication
#1. Element-wise mult(scalar)
#2. Matrix multiplication(dot product)
#There are two main rules matrix mult needs to fit
#Inner dimension is the same as the outer dimension of the target matrix
tensor = torch.tensor([[1,2],[3,4]])
print(tensor*tensor)
matResult = torch.matmul(tensor,tensor)
print(matResult)
#%%
#Note using tensor library is much much more faster than using conventional mult
#torch.mm = torch.matmul
#To manipulate shape issues, we can use transpose
#Note the Inner dimension of two matrix must match,we can use Transpose to assist us
print(tensor.mT)

#%%
#Tensor aggregation
#Min,max,mean and sum of tensors
#Returning from large amount of number to 1 element.
#finding the min
x = torch.arange(1,11,2)
print(x)
print(torch.min(x))
#Notice that now data type error occurs, so we must change the data type
#print(torch.mean(x))
#Changing the data type using .type(dtype) so that function works
print(x.type(torch.float32).mean())
#Sum
print(torch.sum(x))

#%%
#Finding the arg(min), the postion of min in a tensor -> return index position of
#target tensor where minimum occurs
print(x.argmax())


# %%

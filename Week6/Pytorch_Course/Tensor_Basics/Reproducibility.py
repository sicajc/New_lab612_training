#%%
import torch

#In short how a Neural Network learns:
#Starting with random numbers -> tensor operations -> update random numbers to try and make them better representation

#To reduce the randomness in Neural Network comes the conecpt of random seed

random_tensor_A = torch.rand(3,4)
random_tensor_B = torch.rand(3,4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)

#%%
#setting the random seed
RANDOM_SEED = 42

#Uses randomness in python
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
#With this code, this can be used to generate same random number even if run on other platform
 
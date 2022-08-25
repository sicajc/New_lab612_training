#%%
import torch
#Indexing in numpy is same as indexing in pytorch

x = torch.arange(1,10).reshape(1,3,3)
print(x)


#Indexing
print(x[0])

#Index at middle bracket
print(x[0][0])

print(x[0][0][0])
#%%
#get all values of 0th and 1st dimensions but only index 1 of n2d dimension
print(x[:,:,1])

print(x[:,1,1])

#%%

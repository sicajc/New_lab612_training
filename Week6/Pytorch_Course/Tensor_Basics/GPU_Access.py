#%%
import torch
#Running tensors and Pytorch objects on the GPUs
#Checking GPU access from pytorch


torch.cuda.is_available()
#%%
#Setting device agnostic code
#Run on GPU if available !
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
#%%
#Testing
tensor = torch.tensor([1,2,3])

print(tensor,tensor.device)

# %%
#Switching to gpu
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)
# %%
#If tensor on gpu, we cannot transform tensor to numpy!
#To fix the gpu tensor numpy issue, must convert it to CPU
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu) #As numpy
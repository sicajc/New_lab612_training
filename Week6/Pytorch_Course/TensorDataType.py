import torch

#float32 tensor
#datatpye of tensor are float16 or float32!
#Data type is extremely important, it is related to precision in computer
#Device:
#Requires_grad:
float_32_tensor = torch.tensor([3.0,6.0,9.0],dtype = torch.float16,device = None,requires_grad=False)
print(float_32_tensor)

"""An image is nothing but a numpy array,so this is very important"""
#%%
import numpy as np

a = [1,2,3,4,5]
c = np.array(a) #Converting into numpy array
d = 2*c #Increase the brightness of the image
print(f'd = {d}')

# %%
#Arrays of different sizes cannot be added together
a = np.array([1,2,3,4,5])
b = np.array([6,7,8])
c = np.array([9,10,11,12,13])

print(a+b)
print(a+c)

#%%
x = np.array([[1,2],[3,4]]) #Standard using int32
y = np.array([[3.2,4],[5,6]],dtype=np.float64) #Converting value in array into float64
z = y/2
#%%
from skimage import io
img1 = io.imread("images/testing_image.jpeg")
print(type(img1))

import numpy as np
a = np.ones((3,3)) #3x3 array of 1
b = np.zeros((3,3)) #3x3 array of 0
c = np.ones_like(img1) #This creates the array of 1 with the size of the input image(array),useful in later use

n = np.random.random((3,3)) #This creates 3x3 random array which resembles noises


a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) 
print(np.shape(a))

b = a[:2] # b is a subset of 2 , first 2 rows of a

c = a[:2,1:3] # c is first 2 rows's column from 1~3
#%%
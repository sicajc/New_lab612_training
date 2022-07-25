"""Plotting using matplotlib for more analysus details"""
#This line shall be remembered
#%%
from matplotlib import pyplot as plt
import cv2

plt.plot() #This gives a canvas for the plot
#%%

"""Easy example"""
x = [1,2, 3 ,4, 5]
y = [4,7,2,1,8]

"""this plots the image out"""
plt.plot(x,y)
plt.show()
#%%

"""We can also use this on numpy array"""
"""Now read image in format and we can print out the intensity or histogram of an image"""
##Better use Absolute Path s.t. it would not read the wrong image
gray_img = cv2.imread("C:/Users/HIBIKI/Desktop/Lab612_training/Week1/images/testing_image.jpeg",0)

plt.imshow(gray_img,cmap = "gray")
plt.show()

#This flattens out the 2d arrays into 1d arrays then plot the intensity of the image pixels
plt.hist(gray_img.flat,bins = 100,range = (0,150))
plt.show()

"""Different plotting styles, for more infos search for documentation"""
plt.plot(x,y,'bo') #Blue dots
plt.show()

plt.plot(x,y,'r--')
plt.axis([0,6,0,50]) #This defines the range you want the plotting to display
plt.show() #red dots

"""pyplot actually would display all the plt together"""
plt.plot(x,y,linewidth = 5.0) #This enbales thicker line


"""#%% Enables Interactive Windows,used to extract certain part of script for testing"""
#%%
"""For standard way of plotting simply copy the code from others then modify it"""
"""Using subplot is important for image analysis"""
plt.figure(figsize=(12,6))
#Linear plot
plt.subplot(121) #This means 1 row 2 cols in col 1
plt.plot(x,y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)

#Log
plt.subplot(122)
plt.plot(x,y)
plt.yscale('log')
plt.title('log')
plt.grid(True)

plt.show()

"""In the end, we can actually save the plot"""
plt.savefig("plot.jpeg") #.. takes the directory 1 level lighter
# %%

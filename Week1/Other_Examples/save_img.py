"""Saving file using skimage"""
from skimage import io, img_as_float
from skimage import filters
#In skimage is RGB format
#in OPENCV is GBR format
img = io.imread("images/testing_image.jpeg")
#This is a numpy array

print(f"img is of shape {img.shape} \n img is of array\n {img}")
#print(img.shape) #y,x,c(number of channels)

#Apply the gaussian filters
#Size changing from uint8 into float64 after applying filter
gaussian_img = filters.gaussian(img,sigma=3)

#Saving the image
io.imsave("saved_images/saved_using_skimage_after_gaussian_filter.jpg",gaussian_img)
io.imsave("saved_images/saved_using_skimage_after_gaussian_filter.tif",gaussian_img)

"""Note you get blurred image after the filter due to passing through the gaussian filter, however the format is saved as
float64 instead of uint8 so we shall convert it to the correct format later"""

from skimage import img_as_ubyte
#This converts the image from any format back to uint8 format
gaussian_img_8bit = img_as_ubyte(gaussian_img)
io.imsave("saved_images/saved_using_skimage_after_gaussian_filter.tif",gaussian_img_8bit)

##############################################################
"""Then saving file using OPENCV """
import cv2
#For cv2 you had better convert your image to unit8 format first before saving it the way of saving an image
#This would crash if you save in wrong format
cv2.imwrite("saved_images/saved_using_opencv_without_uint8_2_float64.jpg",gaussian_img)

#Remember to convert in into OPENCV BGR format if you want to stored using opencv
cv2.imwrite("saved_images/saved_using_opencv_with_uint8_2_float64_but_NO_RGB_CON.jpg",gaussian_img_8bit)

gaussian_rgb2bgr = cv2.cvtColor(gaussian_img_8bit,cv2.COLOR_RGB2BGR)
cv2.imwrite("saved_images/saved_using_opencv_with_conversion_RGB2BGR_CON.jpg",gaussian_rgb2bgr)

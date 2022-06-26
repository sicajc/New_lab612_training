from skimage import io, img_as_float
import cv2
#In skimage is RGB format
img = io.imread("C:/Users/HIBIKI/Desktop/Lab612_training/Week1/images/testing_image.jpeg")

print(f"img is of shape {img.shape} \n img is of array\n {img}")
#print(img.shape) #y,x,c(number of channels)

img2 = img_as_float(img) #scales images between 0 ~ 1
#then converting image back to ubyte array
print("Img2 is")
print(img2)

image_8bit = img_as_float(img2)
print("Image_8bit is")
print(image_8bit)

#Reading image using opencv
#Open CV reads images as BGR instead of RGB!!!
grey_img = cv2.imread("C:/Users/HIBIKI/Desktop/Lab612_training/Week1/images/testing_image.jpeg",0) #Read into grey_Scale
colour_img = cv2.imread("C:/Users/HIBIKI/Desktop/Lab612_training/Week1/images/testing_image.jpeg",1) #Read as Color images

#Converting BGR in opencv format to RGB , this must be memorised
img_opencv = cv2.cvtColor(colour_img,cv2.COLOR_BGR2RGB)

#All these images are saved as numpy arrays for us to process later on

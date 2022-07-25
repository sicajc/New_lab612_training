"""This process enables immediate feedback, but for further anlysis, better use commercial applications"""
from skimage import io, img_as_float
#We use matplotlib to display immediate feedback
from matplotlib import pyplot as plt

#First read the image file in
img = io.imread("images/testing_image.jpeg")
"""Display image immediately"""
#io.imshow(img)
#plt.show()

"""Or simply use plt.imshow(img) + plt.show()"""
#plt.figure(figsize=(10, 10)) #This can adjust the size we want to display
#plt.imshow(img)
#plt.show()
img_original = plt.imshow(img)

"""matplotlib actually enables a lot of showing features for images"""
"""cmap applys to grey scale images which applys to grey scale images only"""
img_gray = io.imread("images/testing_image.jpeg",as_gray=True)
#If you dont put anything i.e. io.imread("images/testing_image.jpeg"), then what we read in is array uint8 not float
#as_gray True become float64

"""cmap allows you to make clearer view of your image for analysis purposes"""
img_hot = plt.imshow(img_gray,cmap = "hot")
img_jet = plt.imshow(img_gray,cmap = "jet")

"""Multiple plots in pyplot"""
fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img_gray, cmap='hot')
ax1.title.set_text('1st')

ax2 = fig.add_subplot(2,2,2)
ax2.imshow(img_gray, cmap='jet')
ax2.title.set_text('2nd')

ax3 = fig.add_subplot(2,2,3)
ax3.imshow(img_gray, cmap='gray')
ax3.title.set_text('3rd')

ax4 = fig.add_subplot(2,2,4)
ax4.imshow(img_gray, cmap='nipy_spectral')
ax4.title.set_text('4th')
plt.show()

###################################################
"""OPENCV IS MORE CONVENEIENT IN DISPLAYING IMAGES"""
import cv2
#You might easily get erros if you dont use full path!
gray_img = cv2.imread("images/testing_image.jpeg",0)
color_img = cv2.imread("images/testing_image.jpeg",1)

"""Rememver to convert from RGB2BGR if you read using skimage but want to output the image using openCV"""
img_RGB2BGR = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
cv2.imshow("Img from skimage",img_RGB2BGR)

cv2.imshow("Colour img from opencv",color_img)
cv2.imshow("gray_img",gray_img)

##These must be added whenever using imshow
cv2.waitKey(0) #Display for 3000ms, then close the window
cv2.destroyAllWindows()

#%%
import cv2
from matplotlib import cm, pyplot as plt
img = cv2.imread('../images/testing_image.jpeg')

plt.imshow(img) #OPENCV reads image as BGR NOT RGB
plt.show()

resized = cv2.resize(img,None,fx = 2,fy = 2,interpolation= cv2.INTER_CUBIC)

cv2.imshow("Resized pic",resized)
cv2.waitKey(1000) #If colors means something to you in opencv, play close attention
cv2.destroyAllWindows()

# %%

print("Top left",img[0,0]) #Top left pixel
top_left_pixel = img[0,0]

# %%
"""Then we want to extract certain channels out from images RGB"""
"""This is splitting channels in an numpy array of view"""
blue = img[:,:,0]
green = img[:,:,1]
red = img[:,:,2]

cv2.imshow("blue pic",blue)
cv2.imshow("green",green)
cv2.imshow("red",red)
cv2.waitKey(1000) #If colors means something to you in opencv, play close attention
cv2.destroyAllWindows()

#%%
import cv2
from matplotlib import pyplot as plt
""" This can be done easily using opencv function"""
img = cv2.imread('../images/testing_image.jpeg')
b,g,r = cv2.split(img)
cv2.imshow("Resized pic",b)

"""We can also merge images together s.t. using cvmerge"""
img_merged = cv2.merge((b,g,r))
cv2.imshow("Merged image",img_merged)

"""Display the edges of the image"""
edges = cv2.Canny(img,100,200)
cv2.imshow("Edged pic",edges)

cv2.waitKey(5000) #If colors means something to you in opencv, play close attention
cv2.destroyAllWindows()

# %%

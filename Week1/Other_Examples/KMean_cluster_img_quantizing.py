#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
#Read the documentation of K mean carefully to know how to use the function
img =cv2.imread('images/testing_image.jpeg') #CV2 read in array as BGR

#For KMeans we have to flatten the array
img_reshape = img.reshape((-1,3)) # -1 flatten out the array entry, 3 is the element size of each entry

#Note In KMeans documentation np.float32 is used, so change the data type
img_reshape = np.float32(img_reshape)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0) #The criteria for Kmeans

#Clusters
n_colors = 8
#Compactness, label and center
ret,label,center = cv2.kmeans(img_reshape,n_colors,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center) #To enable image disaply , must convert it back to uint8
rescale_img = center[label.flatten()] #Flatten out the label
#Look at the shape of original image then reshape the processed image into the same shape
reshape_img = rescale_img.reshape((img.shape))
cv2.imshow("Quantized Image",reshape_img)
cv2.waitKey(0) #Display for 3000ms, then close the window
cv2.destroyAllWindows()
#%%
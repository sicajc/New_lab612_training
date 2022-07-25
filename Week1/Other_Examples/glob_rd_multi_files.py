"""Glob library is used for reading multiple files and saving multiple images"""
#%%
import cv2
import glob

file_list = glob.glob('saved_images/*.*') #Returns a list of images, a list of 2d arrays
print(file_list)

#%%
#loading each file at a time
my_list = [] #Store the image into a new list

path = "saved_images/*.*"
img_number = 1 #Start an iterator for image number!

for file in glob.glob(path): #glob.glob() used to extract multiple images
    print(file)
    a = cv2.imread(file) #Reading the image file into the list
    my_list.append(a) #Creating a list of images,notice my_list is a collection of arrays
    #Process each image -chaing from BGR to RGB
    c = cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
    cv2.imwrite("test_images/Color_image"+str(img_number)+".jpg",c)
    img_number += 1 #This iterater helps us label the images for further use
    cv2.imshow(f'Color Image number {img_number}',c)
    plt.show()
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

# %%
from matplotlib import pyplot as plt
plt.imshow(my_list[2])
plt.show()

# %%

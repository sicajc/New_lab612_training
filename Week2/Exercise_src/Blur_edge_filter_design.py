#%%
from random import gauss
from skimage import io
from matplotlib import pyplot as plt
import math
import cv2
import numpy as np

#Reading the image in
path = "C:/Users/HIBIKI/Desktop/New_LAB612_Training/Week2/test_images/Lenna.jpg"
raw_img = io.imread(path)
plt.figure(0)
plt.figure(figsize=(10, 10))
plt.title("RAW IMAGE")
plt.imshow(raw_img)
plt.show()

#Average,Gaussian and Median are LPFs
#%%
#Average Filter with nxn kernal
def average_filter(raw_img,kernal_size):
    k_s = kernal_size
    #Number of rows and columns of the image
    m,n,c = raw_img.shape

    #Creating average kernal
    kernal = np.ones([kernal_size,kernal_size],dtype=int)
    sum = np.size(kernal)
    kernal = kernal/sum

    #The place holder for filtered image
    img_filtered = np.zeros_like(raw_img)

    for i in range(int(k_s/2),m-int(k_s/2)):
        for j in range(int(k_s/2),n-int(k_s/2)):
            conv = 0
            for k in range(k_s):
                for l in range(k_s):
                    conv += kernal[k,l] * raw_img[i - int(k_s/2) + k,j - int(k_s/2) + l]

            img_filtered[i,j] = conv

    img_filtered = img_filtered.astype(np.uint8)

    return img_filtered

print("--------------------Testing Average Filter----------------------\n")
kernal_size = 5
avg_img = average_filter(raw_img,kernal_size)
plt.figure(1)
plt.figure(figsize=(10, 10))
plt.title(f"Average Filtered Image of {kernal_size} x {kernal_size} kernal")
plt.imshow(avg_img)
plt.show()

#%%
# Gaussian Filter
# Note 2D gaussian is costy, actually, we would like to exploit seperabilty in Gaussian filter,
# i.e. spliting a 2D gaussian kernal
# into 2 1D gaussian filters
def gaussian_filter(raw_img,sigma,mu,kernal_size):
    img_result = np.zeros_like(raw_img)
    #Number of rows and columns of the image
    m,n,c = raw_img.shape
    k_s = kernal_size

    #Creating a k x k Gaussian Mask
    gaussian_kernal = np.zeros((k_s,k_s),np.float32)
    for i in range(k_s):
        for j in range(k_s):
            norm = math.pow(i-mu,2) + math.pow(j-mu,2)
            gaussian_kernal[i,j] = math.exp(-norm/(2*math.pow(sigma,2)))/2*math.pi*pow(sigma,2)

    sum = np.sum(gaussian_kernal)
    gaussian_kernal = gaussian_kernal/sum

    #Apply the filter to the raw_image
    for i in range(int(k_s/2) , m-int(k_s/2)):
        for j in range(int(k_s/2) , n-int(k_s/2)):
            conv = 0
            for k in range(k_s):
                for l in range(k_s):
                    conv += raw_img[i-int(k_s/2) + k, j- int(k_s/2) + l] * gaussian_kernal[k,l]

            img_result[i,j] = conv

    return img_result

print("----------Testing Gaussian Filter-------\n")
kernal_size = 3
mean = 1.5
var = 4

gau_filtered_img = gaussian_filter(raw_img,var,mean,kernal_size)
plt.figure(2)
plt.figure(figsize=(10, 10))
plt.title(f"Gaussian Filtered Image of {kernal_size} x {kernal_size} kernal")
plt.imshow(gau_filtered_img)
plt.show()

#%%
#Median Filter
#Good at removing salt and pepper noise from the image, a non-linear filter
# However with the cost of blurring in detail, to combat this effect, another filter is proposed
# Bilateral filter is a non-linear filter that combines the features of gaussian and median filter!
def median_filter(raw_img, kernal_size):
    img_result = np.zeros_like(raw_img)

    #Number of rows and columns of the image
    m,n,c = raw_img.shape
    k_s = kernal_size

    #Applying filter to the image
    for i in range(int(k_s/2) , m - int(k_s/2) ):
        for j in range(int(k_s/2) , n - int(k_s/2)):
            #Find the median by extracting components out from the image within the kernal
            kernal = np.zeros((k_s,k_s,3),np.float32)
            for k in range(k_s):
                for l in range(k_s):
                    kernal[k,l] = raw_img[i-int(k_s/2) + k, j- int(k_s/2) + l]

            kernal_r = kernal[:,:,0]
            kernal_g = kernal[:,:,1]
            kernal_b = kernal[:,:,2]

            kernal_r = kernal_r.flatten()
            kernal_g = kernal_g.flatten()
            kernal_b = kernal_b.flatten()

            median_r = sorted(kernal_r)[int(len(kernal_r)/2)]
            median_g = sorted(kernal_g)[int(len(kernal_g)/2)]
            median_b = sorted(kernal_b)[int(len(kernal_b)/2)]

            img_result[i,j] = [median_r,median_g,median_b]

    img_result = img_result.astype(np.uint8)

    return img_result

print("----------Testing Median Filter-------\n")
kernal_size = 3

median_filtered_img = median_filter(raw_img,kernal_size)
plt.figure(3)
plt.figure(figsize=(10, 10))
plt.title(f"Median Filtered Image of {kernal_size} x {kernal_size} kernal")
plt.imshow(median_filtered_img)
plt.show()

#%%
#Sobel Filter


#%%
#Canny Filter



#%%
#Print the result out onto a single plot
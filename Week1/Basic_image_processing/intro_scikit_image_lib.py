#%%
from matplotlib import cm, pyplot as plt

from skimage import io,color
from skimage.transform import rescale,resize,downscale_local_mean

img = io.imread("../images/testing_image.jpeg",as_gray=True)

io.imshow(img) #note this img is a nparray of float64 between 0 and 1
#as_gray = True, it normalised to 0~1 but if remove True, it is uint8
plt.show()

#%%
"""Rescaling the img"""
#Note usually rescaling is a better option than resizing,resize is usually used
img_rescaled = rescale(img,1/4,anti_aliasing=False)
io.imshow(img_rescaled)
plt.show() #Notice from variable explorer, the image is rescaled

img_resized = resize(img,(200,200),anti_aliasing=False)#Resize

plt.imshow(img_resized,cmap = 'hot')
plt.imshow(img_rescaled,cmap = 'hot')

#%%
from matplotlib import cm, pyplot as plt

from skimage import io,color
from skimage.filters import gaussian,sobel #A variety of filter is presented

img = io.imread("../images/testing_image.jpeg")
plt.imshow(img)
#Gaussian image provides a blurring effect
gaussian_img_skimage = gaussian(img,sigma=1,mode='constant',cval = 0.0)
plt.imshow(gaussian_img_skimage)
plt.show()

img_gray = io.imread("../images/testing_image.jpeg",as_gray = True)
sobel_img = sobel(img_gray)
plt.imshow(sobel_img,cmap = 'gray') #Note sobel filter only eats gray image
plt.show()
# %%

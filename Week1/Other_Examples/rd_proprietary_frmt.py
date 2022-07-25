"""
Reading proprietary images i.e. czi or tif files which contains more infos than normal images i.e. jpg and png files
"""
#pip install tifffile
#pip install czifile
#tiffile library is needed to read these proprietary images
import tifffile
import czifile
import numpy as np

#RGB images
img = tifffile.imread("images/screenshot_testing.tif")
print(np.shape(img)) #(1080, 3840, 3) (different images slices, , )

#3d images
img1 = tifffile.imread("images/screenshot_testing.tif")
print(np.shape(img1))

#time series images
img2 = tifffile.imread("images/screenshot_testing.tif")
print(np.shape(img2))
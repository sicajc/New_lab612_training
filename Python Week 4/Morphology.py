import itertools
from skimage import io,color
from matplotlib import pyplot as plt
import math
import cv2
import numpy as np

def dilation():
    #Using 3x3 SE, reflect B about its Origin then shift it by Z, this is as mathematical expression
    #Only if all pixel doesn't match with result after Convolving with SE puts a 0 else puts 1


    return

def erosion():
    #Using 3x3 SE
    #If all pixel matches with the result after img Convoling with SE, puts 1 else puts 0.


    return


def opening():
    #First perform erosion then perform dilation



    return


def closing():
    #First perform dilation then perform erosion


    return


def boundaryExtraction():
    #1. Perform erosion of the input image
    #2. Subtract the eroded image from the original Image


    return
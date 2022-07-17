from igraph import *
import heapq
from matplotlib import pyplot as plt
import cv2
import numpy as np

class HuffmanCoding:
    def __init__(self, path):
        self.path = path
        self.heap = []
        self.codes = {}


    class HeapNode:
        def __init__(self,intensity,freq):
            self.intensity = intensity
            self.freq = freq
            self.left = None
            self.right = None

    #Compression function
    def CreateFrequencyDictionary(self,img):
        frequency = {}
        for intensity in img:
            if intensity not in img:
                frequency[intensity] = 0
            frequency[intensity] += 1

        return frequency

    def CreateMinHeap(self, frequencyDictionary):
        for intensity in frequencyDictionary:
            node = self.HeapNode(intensity, frequencyDictionary[intensity])
            heapq.heappush(self.heap,node)

    def GenerateHuffmanTree(self):
        root = heapq.heappop(self.heap)
        current_code = ""

        self.GenerateSubHuffmanTree(root,current_code)

    def GenerateSubHuffmanTree(self,root,current_code):
        if root is None:
            return
        elif(root.intensity != None):
            self.codes[root.intensity] = current_code
            return

        self.GenerateSubHuffmanTree(root.left, f"{current_code}0")
        self.GenerateSubHuffmanTree(root.right, f"{current_code}1")

    def PrintEncodedDictionary(self):
        self.PlotFrequency_EncodedBitsChart()
        self.OutputDictionary()
        self.VisualiseHuffmanTree()
        self.OutputFile()

    def PlotFrequency_EncodedBitsChart(self):
        return

    def VisualiseHuffmanTree(self):
        return

    def OutputFile(self):
        return

    def Compression(self):
        rawImg = cv2.imread(self.path , 0)
        frequencyDictionary = self.CreateFrequencyDictionary(rawImg)
        self.CreateMinHeap(frequencyDictionary)
        self.GenerateHuffmanTree()
        print("Image Compressed\n")
        self.PrintEncodedDictionary()


print("----------Testing Huffman Enocding-------\n")
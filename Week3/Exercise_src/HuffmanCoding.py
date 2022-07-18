#%%
import json
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
        self.frequencyDict = {}


    class HeapNode:
        def __init__(self,intensity,freq):
            self.intensity = intensity
            self.freq = freq
            self.left = None
            self.right = None

        #These are needed for heap comparison
        def __lt__(self,other):
            return self.freq < other.freq

        def __eq__(self,other):
            if(other == None):
                return False
            if(not isinstance(other)):
                return False
            return self.freq == other.freq

    #Compression function
    def CreateFrequencyDictionary(self):
        rawImg = cv2.imread(self.path,0)

        img = rawImg.ravel()

        for intensity in img:
            if intensity not in self.frequencyDict:
                self.frequencyDict[intensity] = 0

            self.frequencyDict[intensity] += 1

        return self.frequencyDict

    def CreateMinHeap(self):
        for intensity in self.frequencyDict:
            node = self.HeapNode(intensity, self.frequencyDict[intensity])
            heapq.heappush(self.heap,node)

    def GenerateHuffmanTree(self):

        while(len(self.heap) > 1):

            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = self.HeapNode(None,node1.freq + node2.freq)

            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap,merged)

        root = heapq.heappop(self.heap)
        current_code = ""

        self.EncodingHuffmanTree(root,current_code)

    def EncodingHuffmanTree(self,root,current_code):
        if root is None:
            return

        if(root.intensity != None):
            self.codes[root.intensity] = current_code
            return

        self.EncodingHuffmanTree(root.left, f"{current_code}0")
        self.EncodingHuffmanTree(root.right, f"{current_code}1")

    def PrintEncodedDictionary(self):
        self.PlotFrequency_EncodedBitsChart()
        self.VisualiseHuffmanTree()
        self.OutputFile()
        return

    def PlotFrequency_EncodedBitsChart(self):
        print(f"Frequency Dictionary: \n{self.frequencyDict}\n")
        print(f"Code: \n{self.codes}\n")

    def VisualiseHuffmanTree(self):
        return

    def OutputFile(self):
        return

    def ChangeType_OrderOfDictionary(self):
        self.codes = dict(sorted(self.codes.items()))
        test.codes = {str(key):str(encoded_str) for key,encoded_str in test.codes.items()}


    def Compression(self):
        self.CreateFrequencyDictionary()
        self.CreateMinHeap()
        self.GenerateHuffmanTree()

        print("Image Compressed\n")
        self.ChangeType_OrderOfDictionary()
        self.PlotFrequency_EncodedBitsChart()
        return

print("----------Testing Huffman Enocding-------\n")

path1 = "C:/Users/HIBIKI/Desktop/New_LAB612_Training/Week3/lena.bmp"
test = HuffmanCoding(path1)
test.Compression()

keys = list(test.codes.keys())
encoded_bits = list(test.codes.values())

# plt.bar(range(len(test.codes)), encoded_bits, tick_label=keys)
# plt.show()

with open('codedResult.txt','w') as file:
    file.write(json.dumps(test.codes))

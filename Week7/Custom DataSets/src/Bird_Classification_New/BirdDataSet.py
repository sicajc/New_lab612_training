import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io


class BirdsDataset(Dataset):#<- subclass of being a DataSET
    def __init__(self,dataset,csv_file,root_dir,transform = None):
        self.csv = pd.read_csv(csv_file)
        self.annotations = self.csv.loc[self.csv["data set"]==dataset].reset_index()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self,index):
        #<- iloc index location, take the item out of the index
        #<- iloc[0,3] means taking the item from index  0 and index 3
        class_index = self.annotations["class index"][index]
        class_index = torch.tensor(class_index)

        img_location =self.root_dir + "./" + self.annotations["filepaths"][index]

        image = io.imread(img_location)/255

        if self.transform:
            image = self.transform(image)

        return (image,class_index)

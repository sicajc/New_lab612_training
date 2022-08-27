import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io


class BirdsDataset(Dataset):#<- subclass of being a DataSET
    def __init__(self,csv_file,root_dir,transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self,index):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0]) #<- iloc index location, take the item out of the index
        #<- iloc[0,3] means taking the item from index  0 and index 3

        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform:
            image = self.transform(image)

        return (image,y_label)

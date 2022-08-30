import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os
import pandas
import torch

from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


class CustomDataset(Dataset):
    def __init__(self, dataset="train", transform=torchvision.transforms.Resize(size=(224, 224)), target_transform=None):

        self.image_idx_loc = []
        self.imageClass = []

        self.dataFrame = pandas.read_csv('./birds.csv')
        self.dataFrame = self.dataFrame.loc[self.dataFrame["data set"] == dataset].reset_index()
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, idx):
        img_loc = self.dataFrame['filepaths'][idx]
        img_cls = self.dataFrame['class index'][idx]

        img_path = "./" + img_loc
        img = read_image(img_path) / 255

        if self.transform:
            img = self.transform(img)

        img = img.to(device)
        img_cls = torch.tensor(img_cls).to(device)

        return img, img_cls


class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        #
        # Define network structure
        #
        self.conv1 = self.ConvLayer(3, 96, (11, 11), 4, (2, 2), 2, pad=0, pooling=True)
        self.conv2 = self.ConvLayer(96, 256, (5, 5), 1, (2, 2), 2, pad=2, pooling=True)
        self.conv3 = self.ConvLayer(256, 384, (3, 3), 1)
        self.conv4 = self.ConvLayer(384, 384, (3, 3), 1)
        self.conv5 = self.ConvLayer(384, 256, (3, 3), 1, (2, 2), 2, pooling=True)
        self.linear1 = self.LinearLayer(9216, 4096)
        self.linear2 = self.LinearLayer(4096, 4096)
        self.linear3 = torch.nn.Linear(4096, 400)

        self.flatten = torch.nn.Flatten()

    def forward(self, in_feature):
        out_feature = self.conv1(in_feature)
        out_feature = self.conv2(out_feature)
        out_feature = self.conv3(out_feature)
        out_feature = self.conv4(out_feature)
        out_feature = self.conv5(out_feature)
        out_feature = self.flatten(out_feature)
        out_feature = self.linear1(out_feature)
        out_feature = self.linear2(out_feature)
        out_feature = self.linear3(out_feature)

        return out_feature

    def ConvLayer(self, iC, oC, cK, cS, pK=None, pS=None, pad=1, pooling=False):
        layer = OrderedDict([
            ('conv', torch.nn.Conv2d(in_channels=iC,out_channels=oC,kernel_size=cK,stride=cS,padding=pad)),
            ('relu', torch.nn.ReLU())
        ])
        
        if pooling:
            layer['pooling'] = torch.nn.MaxPool2d(kernel_size=pK, stride=pS)

        return torch.nn.Sequential(layer)

    def LinearLayer(self, iN, oN):
        layer = OrderedDict([
            ('fc', torch.nn.Linear(iN, oN)),
            ('relu', torch.nn.ReLU()),
            ('dropout', torch.nn.Dropout(p=0.5))
        ])
        
        return torch.nn.Sequential(layer)


# Define training parameters
batch_size = 64
epoch = 100
lr = 1e-4

loss_point = []
loss_window = []

# Create model
model = AlexNet().to(device)

# Loading dataset and setting data loader
train_dataset = CustomDataset(dataset="train")
valid_dataset = CustomDataset(dataset="valid")
test_dataset = CustomDataset(dataset="test")

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
                              
valid_dataloader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=True)


test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True)

# Define loss function and optimization
loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

#
# Create training loop
#
print("(training...)")
for e in range(epoch):
    for batch, (image, label) in enumerate(train_dataloader):
        # Compute prediction and loss
        predict = model.forward(image)
        loss = loss_fn(predict, label)

        # Optimize the model
        loss.backward()
        optim.step()
        optim.zero_grad()

        if batch > 10 and e == 0:
            loss_window.pop(0)
            loss_window.append(loss.item())
            loss_point.append(sum(loss_window) / len(loss_window))
        else:
            loss_window.append(loss.item())
            loss_point.append(sum(loss_window) / len(loss_window))

        if batch % 10 == 0:
            print("Epoch : {} | Batch : {}/{} | Loss : {:0.4f}".format(e, batch * batch_size, len(train_dataset), loss.item()))
    
    print("(valid...)")
    with torch.no_grad():
        predictCorrect = 0
        for image, label in valid_dataloader:
            # Compute prediction and loss
            predict = model.forward(image)
            predict = predict.argmax(axis=1)
    
            # Calculate matches
            match = (predict == label)
            predictCorrect += match.sum()
    
        print("Acc : {:2.2f}%".format(100 * predictCorrect / len(valid_dataset)))
#
# Create testing loop
#
print("(test...)")
with torch.no_grad():
    predictCorrect = 0
    for image, label in test_dataloader:
        # Compute prediction and loss
        predict = model.forward(image)
        predict = predict.argmax(axis=1)

        # Calculate matches
        match = (predict == label)
        predictCorrect += match.sum()

    print("Acc : {:2.2f}%".format(100 * predictCorrect / len(test_dataset)))

plt.title("Mean Loss | Window Size = 10")
plt.plot(loss_point)
plt.show()
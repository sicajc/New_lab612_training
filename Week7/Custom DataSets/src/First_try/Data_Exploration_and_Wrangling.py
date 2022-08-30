import os
import glob
import torch


def Data_Wrangling(train_dir,test_dir,valid_dir):
    #Exploring Dataset
    classes = os.listdir(train_dir)
    print(f"Total classes: {classes}")

    #Count the total train,valid,test image
    train_count = 0
    valid_count = 0
    test_count  = 0

    for _class in classes:
        train_count += len(os.listdir(train_dir + _class))
        valid_count += len(os.listdir(valid_dir + _class))
        test_count  += len(os.listdir(test_dir + _class))

    print("Total train images: ",train_count)
    print("Total valid images: ",valid_count)
    print("Total test images: ",test_count)

    #

#%%
import os
import requests
import zipfile
from pathlib import Path


def get_data(name_of_dataset:str,dataURL:str):

    """Download data from the input URL.

    Download the data from the URL specified then unzipping it into target directory

    Args:
        name_of_dataset: str, the name of the dataset to download
        dataURL: str, the online URL of dataset to download
    Returns:
        None
    """
    dataPath = Path("data/")
    imagePath = dataPath / name_of_dataset

    if imagePath.is_dir():
        print(f"{imagePath} directory exists.")
    else:
        print(f"Did not find {imagePath} directory, creating one...")
        imagePath.mkdir(parents=True, exist_ok=True)

    #Download the dataset
    with open(dataPath/name_of_dataset, "wb") as f:
        request = requests.get(dataURL)
        print(f"Downloading {name_of_dataset}")
        f.write(request.content)

    #Unzipping the dataset
    with zipfile.Zipfile(dataPath/name_of_dataset,"r") as zip_ref:
        print(f"Unzipping {name_of_dataset}")
        zip_ref.extractall(imagePath)

    os.remove(dataPath/name_of_dataset)


if __name__ == '__main__':
  #Name of dataset
  DataSet_Name = "archive"
  #Having problem here~ cannot extract the file
  DATA_URL = "https://storage.googleapis.com/kaggle-data-sets/534640/3197090/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220826%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220826T090300Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4609d241ec915c9526e86a2231cb0b0fc3618ec8c25fcf35214a2ff0717f147c533fe85253e200c9ceaeba98a3caf12e638862b75aa0af684c8e5d2b79bfd404c3d3da4e1b16421a2456e0075b0e78edb024afbc8cbc5da081fb357a33bcca8236d313647a7d3e6021d0a29de171390b92f792d836cb2cd58542c583d42be3c1babab206326e5b0937d65c47cd29f9d52fddf5321f3552b1f3c9a769473b0b04dd6896bac4b3cc60a45c18169543222499ced5051ed85df2966119f6d3f1abe3ac25b62de2403fd1418a01449118aa9eb155b2eae3470d8b803febe54390aa861f93bea7a2330272398110cda1c8a8867afcd57b62fb546fce5b5964d4de955f"

  #Getting data
  get_data(DataSet_Name,DATA_URL)
#%%
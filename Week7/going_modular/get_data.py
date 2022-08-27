import os
import requests
import zipfile #<- used to open the zipfile
from pathlib import Path

#Setting up path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

#If image folder doesn't exist download it and prepare it
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)


#Downloading pizza,steak,sushi data
with open(data_path / "pizza_steak_sushi.zip","wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza, steak, sushi data...")
    f.write(request.content)

# Unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...")
    zip_ref.extractall(image_path)

os.remove(data_path / "pizza_steak_sushi.zip")
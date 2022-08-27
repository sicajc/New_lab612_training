import os
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    valid_dir:str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):

  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of CPU cores to use per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
      """

  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)
  valid_data = datasets.ImageFolder(valid_dir, transform=transform)

  #Sampler set
  train_random_sampler = RandomSampler(train_data)
  valid_random_sampler = RandomSampler(valid_data)
  test_random_sampler = RandomSampler(test_data)
  # Get class names
  class_names = train_data.classes
  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      #shuffle=True,
      sampler = train_random_sampler,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      #shuffle=True,
      sampler = test_random_sampler,
      num_workers=num_workers,
      pin_memory=True,
  )
  valid_dataloader = DataLoader(
    valid_data,
    batch_size=batch_size,
    #shuffle=True,
    sampler = valid_random_sampler,
    num_workers=num_workers,
    pin_memory=True,
  )
  return train_dataloader, test_dataloader,valid_dataloader, class_names
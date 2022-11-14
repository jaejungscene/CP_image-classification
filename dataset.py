## functions to create dataset and dataloader

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from PIL import Image
from skimage import io
import pandas as pd

DATA_DIR = "/home/ljj0512/shared/data"

class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_path = self.df["image_path"].iloc[index]
        if self.transform:
            image = self.transform(Image.open(img_path))
        else:
            image = io.imread(img_path) # numpy array로 읽어옴
        label = self.df["label"].iloc[index]
        return [image, label]


class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dir = "/home/ljj0512/private/workspace/CP_urban-datathon_CT/test"
    
    def __len__(self):
        return len(os.listdir(self.dir))

    def __getitem__(self, index):
        img_path = os.listdir(self.dir)[index]
        image = self.transform(Image.open(os.path.join(self.dir,img_path)))
        return image


def create_dataloader(args):
    train_df = pd.read_csv("train_df.csv")
    val_df = pd.read_csv("val_df.csv")
    train_set = CustomDataset(train_df, transform=get_transform("train"))
    val_set = CustomDataset(val_df, transform=get_transform("val"))
    test_set = TestDataset(transform=get_transform("test"))
    # train_set = ImageFolder(root = DATA_DIR+"/Train",
    #                     transform = get_transform("train"))

    # val_set = ImageFolder(root = DATA_DIR+"/Validation",
    #                         transform = get_transform("valid"))
    train_loader = DataLoader(dataset=train_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=4)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4)
    test_loader = DataLoader(dataset=test_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4)
    
    return train_loader, val_loader,  test_loader, 5


def get_transform(param):
    if param == "train":
        transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomAdjustSharpness(sharpness_factor=2),
                        transforms.ToTensor(),
                    ])
    elif param == "val":
        transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomAdjustSharpness(sharpness_factor=2),
                        transforms.ToTensor(),
                    ])
    elif param == "test":
        transform = transforms.Compose([
                        # transforms.RandomAdjustSharpness(sharpness_factor=2),
                        transforms.ToTensor(),
                    ])
    return transform
    

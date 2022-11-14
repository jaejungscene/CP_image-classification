import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
from torch import LongTensor
import pandas as pd 
import typing as ty
import yaml
import numpy as np
from dataset import *
from utils import *
import random
from torchvision.models import  EfficientNet_B6_Weights, EfficientNet_B0_Weights, EfficientNet_B7_Weights
import torchvision.models as models

import wandb
import torch.optim as optim


class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dir = "/home/ljj0512/private/workspace/CP_urban-datathon_CT/test"
    
    def __len__(self):
        return len(os.listdir(self.dir))

    def __getitem__(self, index):
        img_path = sorted(os.listdir(self.dir))[index]
        image = self.transform(Image.open(os.path.join(self.dir,img_path)))
        return image, img_path

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def model_infer():
    seed_everything(0) # Seed 고정

    test_set = TestDataset(transform=get_transform("test"))
    dl_test = DataLoader(dataset=test_set,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4)

    submit = pd.read_csv("/home/ljj0512/private/workspace/CP_urban-datathon_CT/1001_sample_submission.csv")
    model_path = "/home/ljj0512/private/workspace/CP_urban-datathon_CT/efficientnet-b6_100.pt"

    model = EfficientNet.from_pretrained('efficientnet-b6', num_classes = 5)
    model.load_state_dict(torch.load(model_path))       
    model.cuda()
    model = nn.DataParallel(model)

    model.eval()
    model_preds = []
    D = {'ILD': 0, 'Lung_Cancer': 1, 'Normal': 2, 'pneumonia': 3, 'pneumothorax': 4}
    T = [2,1,0,3,4]
    with torch.no_grad():
        for img, path in dl_test:
            img = img.float().cuda()
            model_pred = model(img)
            model_pred = model_pred.squeeze(1).to('cpu')
            # prediction = np.concatenate((prediction, model_pred.detach().cpu().numpy()), axis = 1)
            # print(prediction.shape)
            temp = model_pred.argmax(1).detach().cpu().numpy().tolist()

            model_preds += [T[temp[0]]]
    print(len(model_preds))
    submit["result"] = model_preds
    submit.to_csv('/home/ljj0512/private/workspace/CP_urban-datathon_CT/submit13.csv', index=False)


# python main.py --action train --seed 0 --model efficientnet-b6 --epochs 100 --batchsize 16 --savepath savemodel
from efficientnet_pytorch import EfficientNet
from torchvision import models

def load_model(args):
    if args.model == "efficientnet-b0":
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes = 5)
        return model
    elif args.model == "efficientnet-b5":
        model = EfficientNet.from_pretrained('efficientnet-b5', num_classes = 5)
        return model
    elif args.model == "efficientnet-b6":
        model = EfficientNet.from_pretrained('efficientnet-b6', num_classes = 5)
        return model
    elif args.model == "efficientnet-b7":
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes = 5)
        return model
    else:
        return "Load Model ERROR"

if __name__ == '__main__':
    model_infer()
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
import torch
import pandas as pd
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import get_transform
from PIL import Image
from torchvision.models import  EfficientNet_B6_Weights, EfficientNet_V2_M_Weights


class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dir = "/home/ljj0512/private/workspace/CP_urban-datathon_CT/test"
    
    def __len__(self):
        return len(os.listdir(self.dir))

    def __getitem__(self, index):
        img_path = sorted(os.listdir(self.dir))[index]
        image = self.transform(Image.open(os.path.join(self.dir,img_path)))
        return image


def main():
    num_class = 5
    path = "/home/ljj0512/private/workspace/CP_urban-datathon_CT/weights"
    test_set = TestDataset(transform=get_transform("test"))
    test_loader = DataLoader(dataset=test_set,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4)
    path = "/home/ljj0512/private/workspace/CP_urban-datathon_CT/log/08:31:19_eff-b6_f1-0.99_acc-99.7/checkpoint.pth.tar"

    
    checkpoint = torch.load(path)
    model = models.efficientnet_b6()
    model.classifier[1] = nn.Linear(in_features=2304, out_features=5, bias=True)
    model.load_state_dict(checkpoint['state_dict'])    
    # model = nn.DataParallel(model)
    inference(model, test_loader)


def inference(model, test_loader):
    model.cuda()
    model.eval()
    preds = []
    submit = pd.read_csv("./1001_sample_submission.csv")
    with torch.no_grad():
        for img in (test_loader):
            img = img.cuda()
            output = model(img)
            _, predicted = torch.max(output.data, dim=1)
            preds += predicted.cpu().tolist()
    
    submit['result'] = preds
    submit.to_csv('./1001_submission.csv', index=False)
    print("inference finish")


if __name__ == '__main__':
    main()
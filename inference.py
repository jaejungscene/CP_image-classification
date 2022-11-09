import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "9"
import torch
import pandas as pd
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import get_transform
from PIL import Image

class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dir = "/home/ljj0512/shared/data/test"
    
    def __len__(self):
        return len(os.listdir(self.dir))

    def __getitem__(self, index):
        img_path = os.listdir(self.dir)[index]
        image = self.transform(Image.open(os.path.join(self.dir,img_path)))
        return image

def main():
    test_set = TestDataset(transform=get_transform("test"))
    test_loader = DataLoader(dataset=test_set,
                            batch_size=16,
                            shuffle=False,
                            num_workers=4)
    checkpoint = torch.load("/home/ljj0512/private/workspace/CP_urban-datathon_CT/log/2022-11-09 14:02:59/checkpoint.pth.tar")
    model = models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=5, bias=True)
    model.load_state_dict(checkpoint["state_dict"])
    model = nn.DataParallel(model).cuda()
    inference(model, test_loader)


def inference(model, test_loader):
    model.cuda()
    model.eval()
    preds = []
    submit = pd.read_csv("./sample_submission.csv")
    with torch.no_grad():
        for img in (test_loader):
            img = img.cuda()
            output = model(img)
            _, predicted = torch.max(output.data, dim=1)
            preds += predicted.cpu().tolist()
    
    submit['result'] = preds
    submit.to_csv('./submission.csv', index=False)


if __name__ == '__main__':
    main()
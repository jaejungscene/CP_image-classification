import os
from args import get_args_parser
args = get_args_parser().parse_args()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

import time
import dataset
import numpy as np
import wandb
import random
from datetime import datetime
from log import save_checkpoint, printSave_one_epoch, printSave_start_condition, printSave_end_state
from utils import accuracy, adjust_learning_rate, AverageMeter, get_learning_rate
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torchvision.models import  EfficientNet_B6_Weights, EfficientNet_B0_Weights, EfficientNet_B7_Weights, EfficientNet_V2_M_Weights
from optimizer import get_optimizer_and_scheduler
from sklearn import metrics
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

result_folder_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
best_acc = -1
best_f1 = -1

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(0) # Seed 고정


def create_model(args, numberofclass):
    # model = models.efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
    if args.model == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=numberofclass, bias=True)
    elif args.model == "efficientnet_b6":
        model = models.efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=2304, out_features=numberofclass, bias=True)
    elif args.model == "efficientnet_b7":
        model = models.efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=2560, out_features=numberofclass, bias=True)
    elif args.model == "efficientnet_v2_m":
        model = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=numberofclass, bias=True)
    # elif args.model == "resume":
    #     print("check")
    #     model = models.efficientnet_b6()
    #     model.classifier[1] = nn.Linear(in_features=2304, out_features=numberofclass, bias=True)
    #     checkpoint = torch.load("/home/ljj0512/private/workspace/CP_urban-datathon_CT/log/2022-11-10 08:31:16/checkpoint.pth.tar")
    #     model.load_state_dict(checkpoint["state_dict"])
    elif args.model == "resnet18": # test
        model = models.resnet18()
        model.fc = nn.Linear(in_features=512, out_features=numberofclass, bias=True)

    print("=> model:\t'{}'".format(args.model))
    print('=> the number of model parameters: {:,}'.format(sum([p.data.nelement() for p in model.parameters()])))
    return nn.DataParallel(model).cuda()


def run():
    start = time.time()
    global args, best_err1, best_err5, best_f1, best_acc

    train_loader, val_loader, test_loader, numberofclass = dataset.create_dataloader(args)
    model = create_model(args, numberofclass)
    printSave_start_condition(args)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth).cuda()
    optimizer, scheduler = get_optimizer_and_scheduler(model, args, len(train_loader))
    cudnn.benchmark = True

    for epoch in range(0, args.epoch):
        # train for one epoch
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, args)

        # evaluate on validation set
        f1, acc, val_loss = validate(val_loader, model, criterion, epoch, args)

        # remember best prec@1 and save checkpoint
        if best_f1 < f1:
            best_acc = acc
            best_f1 = f1
            save_checkpoint({
                'epoch': epoch,
                'best_f1': best_f1,
                'best_acc': best_acc,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, args)
            
        if args.wandb == True:
            wandb.log({'valid f1 score':f1, 'acc':acc, 'train loss':train_loss, 'validation loss':val_loss})

        print(f'Current best score =>\tf1: {best_f1} , acc: {best_acc}')
        print("-"*100)

    total_time = time.time()-start
    total_time = time.strftime('%H:%M:%S', time.localtime(total_time))
    print(f"train finish (total time: {total_time}")
    inference(model, test_loader)





def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    total = 0
    correct = 0
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        end = time.time()

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        _, predicted = torch.max(output.data, dim=1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        losses.update(loss.item(), input.size(0))
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    acc = 100*correct/total
    print('Epoch: [{0}/{1}]\t'
            'LR: {LR:.6f}\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Accuracy {acc:.4f}({cor}/{total})\t'
            .format(epoch, args.epoch, 
            LR=scheduler.get_lr()[0], batch_time=batch_time, data_time=data_time,
            loss=losses, acc=acc, cor=correct, total=total))
    # printSave_one_epoch(epoch, args, batch_time, data_time, top1, top5, losses)
    # print('* Epoch[{0}/{1}]\t Top 1-err {top1.avg:.3f}\t  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
    #     epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg




def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    pred_labels = []
    true_labels = []
    total = 0
    correct = 0
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        true_labels += target.tolist()

        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        # err1, err5 = accuracy(output.data, target, topk=(1, 5))
        _, predicted = torch.max(output.data, dim=1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        pred_labels += predicted.cpu().tolist()

        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    f1_score = metrics.f1_score(y_true=true_labels, y_pred=pred_labels, average='macro')
    acc = 100*correct/total
    print('Test (on val set): [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Accuracy {acc:.4f}({cor}/{total})\t'
            'F1 {F1:.4f}\t'
        .format(epoch, args.epoch, batch_time=batch_time, loss=losses,
        data_time=data_time, acc=acc, cor=correct, total=total, F1=f1_score))

    # printSave_one_epoch(epoch, args, batch_time, data_time, top1, top5, losses, False)
    # print('* Epoch[{0}/{1}]\t Top 1-err {top1.avg:.3f}\t  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
    #     epoch+1, args.epochs, top1=top1, top5=top5, loss=losses))
    return f1_score, acc, losses.avg


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


if __name__ == '__main__':
    if args.wandb == True:
        wandb.init(project='CP_urban-datathon', name=args.expname, entity='jaejungscene')
    run()

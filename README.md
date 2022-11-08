# self-directed research
- implemented model: resnet, se-resnet, cbam-resnet
- pretrained and fine-tuned model: ViT, DeiT

## Requirements
- Python3
- PyTorch (> 1.0)
- torchvision (> 0.2)

## Train Examples
- CIFAR-10, CIFAR-100: training input size = 32x32 // CNN
```
python train.py --insize 32 --net_type resnet \
--depth 50 --dataset cifar100 --batch_size 64 --optim sgd \
--lr 0.1 --epochs 100 --no-verbose --cuda 0,1 --wandb 1
```

- CIFAR-10, CIFAR-100: training input size = 224x224 // CNN
```
python train.py --insize 224 --net_type resnet \
--depth 50 --dataset cifar100 --batch_size 64 --optim sgd \
--lr 0.1 --epochs 100 --no-verbose --cuda 0,1 --wandb 1
```

- CIFAR-10, CIFAR-100: training input size = 224x224 // pretrained resnet50
```
python train.py --insize 224 --net_type pretrained-resnet \
--depth 50 --dataset cifar100 --batch_size 64 --optim sgd \
--lr 0.1 --epochs 100 --no-verbose --cuda 0,1 --wandb 1
```

- CIFAR-100: training input size = 224x224 // pretrained ViT
```
python train.py --insize 224 --net_type pretrained-deit \
--dataset cifar100 --batch_size 256 --optim sgd \
--lr 0.003 --epochs 100 --no-verbose --cuda 0,1 --wandb 1
```

- CIFAR-100: training input size = 224x224 // pretraind DeiT (knowledge distillation)
```
python train.py --insize 224 --net_type pretrained-deit \
--dataset cifar100 --batch_size 256 --optim sgd \
--distil 1 --distil_type hard \
--lr 0.003 --epochs 100 --no-verbose --cuda 0,1 --wandb 1
```
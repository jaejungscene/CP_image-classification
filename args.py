import argparse
from datetime import datetime
result_folder_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_args_parser():
    parser = argparse.ArgumentParser(description='competition')
    parser.add_argument("--fold", default = 5, type = int)
    parser.add_argument('--model', default='resnet', type=str, help='networktype: resnet')
    parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--epoch', default=100, type=int, metavar='N', help='number of total epochs to run')                    
    parser.add_argument('--cuda', type=str, default='0', help='select used GPU')
    parser.add_argument('--wandb', type=int, default=1, help='choose activating wandb')
    parser.add_argument('--seed', type=str, default=41, help='set seed')
    parser.add_argument('--expname', default=result_folder_name, type=str, help='name of experiment')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='W', help='number of data loading workers (default: 4)')
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--cude', default="0", type=str)

    loss = parser.add_argument_group('loss')
    loss.add_argument('--label_smooth', type=float, default=0.0, help='optimizer name')

    optimizer = parser.add_argument_group('optimizer')
    optimizer.add_argument('--optimizer', type=str, default='adamw', help='optimizer name')
    optimizer.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    optimizer.add_argument('--weight-decay', type=float, default=1e-3, help='optimizer weight decay')
    optimizer.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    optimizer.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999], help='adam optimizer beta parameter')
    optimizer.add_argument('--eps', type=float, default=1e-8, help='optimizer eps')
    optimizer.add_argument('--decay-rate', type=float, default=0.1, help='lr decay rate')

    scheduler = parser.add_argument_group('scheduler')
    scheduler.add_argument('--scheduler', type=str, default='cosinerestarts', help='lr scheduler')
    scheduler.add_argument('--cosine-freq', type=int, default=4, help='cosine scheduler frequency')
    scheduler.add_argument('--restart-epoch', type=int, default=20, help='warmup restart epoch period')
    scheduler.add_argument('--three-phase', action='store_true', help='one cycle lr three phase')
    scheduler.add_argument('--step-size', type=int, default=2, help='lr decay step size')
    scheduler.add_argument('--min-lr', type=float, default=1e-6, help='lowest lr used for cosine scheduler')
    scheduler.add_argument('--milestones', type=int, nargs='+', default=[150, 225], help='multistep lr decay step')
    scheduler.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    scheduler.add_argument('--warmup-scheduler', type=str, default='linear', help='warmup lr scheduler type')
    scheduler.add_argument('--warmup-lr', type=float, default=1e-4, help='warmup start lr')
    scheduler.add_argument('--eta_max', type=float, default=1e-1, help='cosinerestarts max lr')

    parser.add_argument('--distil', type=int, default=0, help='choose whether to do knowledge distillation')
    parser.add_argument('--distil_type', type=str, default='hard', help='choose what type of knowledge distillation')
    # parser.add_argument('--alpha', default=300, type=float,
    #                     help='number of new channel increases per depth (default: 300)')
    # parser.add_argument('--beta', default=0, type=float,
    #                     help='hyperparameter beta')
    # parser.add_argument('--cutmix_prob', default=0, type=float,
    #                     help='cutmix probability')

    return parser
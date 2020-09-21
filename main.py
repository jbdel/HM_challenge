import argparse, os, random
import torch
from torch.utils.data import DataLoader
from models import *
from datasets import *
from train import train
from models import adaboost
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model', type=str, default="PrepareVisualBertModel")
    parser.add_argument('--dataset', type=str, default="HMVisualBertDataset")
    parser.add_argument('--data_path', type=str, default="data/VisualBert")
    parser.add_argument('--use_pretrained_params', type=int, default=0)
    parser.add_argument('--pretrained_params_path', type=str, default="pretrained_params")

    # Training
    parser.add_argument('--output', type=str, default='ckpt/')
    parser.add_argument('--name', type=str, default='exp0/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--lr_base', type=float, default=5e-05)
    parser.add_argument('--eps', type=float, default=1e-08)
    parser.add_argument('--grad_norm_clip', type=float, default=-1)
    parser.add_argument('--eval_start', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=3)
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999999))

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Base on args given, compute new args
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DataLoader
    train_ds = eval(args.dataset)(name="train", args=args)
    dev_ds = eval(args.dataset)(name="dev", args=args)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4)
    eval_loader = DataLoader(dev_ds, args.batch_size, num_workers=4)

    # Test
    # outputs = evaluate_visual_bert(eval_loader=eval_loader, args=args)
    # print(outputs[5])

    net = eval(args.model)(args=args)
    print("Total number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e6) + "M")
    net.model.cuda()

    # Create Checkpoint dir
    if not os.path.exists(os.path.join(args.output, args.name)):
        os.makedirs(os.path.join(args.output, args.name))

    # # Run adaboost
    estimators, estimator_weights, estimator_errors, ensemble_accs = adaboost(base_estimator=net, n_estimators=10, train_loader=train_loader, eval_loader=eval_loader, args=args)
    print(estimator_weights, estimator_errors, ensemble_accs)
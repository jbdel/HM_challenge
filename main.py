import argparse, os, random
import torch
from torch.utils.data import DataLoader
from models import *
from datasets import *
from train import train
import numpy as np
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model', type=str, default="VisualBERT")
    parser.add_argument('--dataset', type=str, default="HatefulMemesFeaturesDataset")
    parser.add_argument('--data_path', type=str, default="data/VisualBert")
    parser.add_argument('--params_path', type=str, default="pretrained_params")

    # Training
    parser.add_argument('--output', type=str, default='ckpt/')
    parser.add_argument('--name', type=str, default='exp0/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=99)
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

    config = OmegaConf.load('facebook.yaml')

    train_ds = eval(args.dataset)(config.dataset_config.hateful_memes, dataset_type="train")
    dev_ds = eval(args.dataset)(config.dataset_config.hateful_memes, dataset_type="val")
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4)
    eval_loader = DataLoader(dev_ds, args.batch_size, num_workers=4)

    # Net

    net = eval(args.model)(config.model_config.visual_bert)
    net.build()
    net.init_losses()
    net.model.cuda()
    print("Total number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e6) + "M")


    # Create Checkpoint dir
    if not os.path.exists(os.path.join(args.output, args.name)):
        os.makedirs(os.path.join(args.output, args.name))

    # # Run training
    eval_accuracies, eval_auroc = train(net, train_loader, eval_loader, args)



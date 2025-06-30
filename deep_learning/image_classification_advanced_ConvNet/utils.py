from pathlib import Path
from typing import Dict, List
import random
import numpy as np
import torch
import torchvision
import torch.nn.init as init
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt

# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Custom learning rate scheduler
def custom_lr_schedule(step):
    if step < 17000:
        return 1.0
    elif step < 19000:
        return 0.1
    elif step < 20000:
        return 0.01
    else:
        return 0.001

# Get learning rate
def get_lr(opt): return opt.param_groups[0]['lr']

# He normal initialization
def init_weights_He_normal(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)
        
# He uniform initialization
def init_weights_He_uniform(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)

# Plotting
def plot_history(hist: Dict[str,List], out_png: Path):
    steps = hist['step']
    plt.figure(figsize=(8,6))
    plt.subplot(3,1,1); plt.plot(steps,hist['train_loss'],label='Train')
    plt.plot(steps,hist['val_loss'],label='Val');plt.legend();plt.title("Loss")
    plt.subplot(3,1,2); plt.plot(steps,hist['train_acc'])
    plt.plot(steps,hist['val_acc']);plt.title("Accuracy")
    plt.subplot(3,1,3); plt.plot(steps,hist['eta']); plt.title("η")
    plt.tight_layout(); plt.savefig(out_png,dpi=300); plt.close()
    
# Data
def loaders(DATASET, AUGMENT, DATA_ROOT, SEED, BATCH_SIZE):
    if DATASET == 'CIFAR10':
        stats = ((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
        tf_train = [T.ToTensor(), T.Normalize(*stats)]
        if AUGMENT:
            tf_train = [T.RandomCrop(32,4),T.RandomHorizontalFlip()]+tf_train
        train_set = torchvision.datasets.CIFAR10(DATA_ROOT,True,download=True,
                                                 transform=T.Compose(tf_train))
        test_set  = torchvision.datasets.CIFAR10(DATA_ROOT,False,download=True,
                                                 transform=T.Compose([
                                                     T.ToTensor(),T.Normalize(*stats)]))
        
    elif DATASET == 'CIFAR100':
        stats = ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        tf_train = [T.ToTensor(), T.Normalize(*stats)]
        if AUGMENT:
            tf_train = [T.RandomCrop(32, 4), T.RandomHorizontalFlip()] + tf_train
        train_set = torchvision.datasets.CIFAR100(DATA_ROOT, True, download=True,
                                                  transform=T.Compose(tf_train))
        test_set = torchvision.datasets.CIFAR100(DATA_ROOT, False, download=True,
                                                 transform=T.Compose([
                                                     T.ToTensor(), T.Normalize(*stats)]))
    val_size = 5000
    train_set, val_set = torch.utils.data.random_split(
        train_set, [len(train_set) - val_size, val_size],
        generator=torch.Generator().manual_seed(SEED))
    
    def loader(ds, shuffle):
        return torch.utils.data.DataLoader(ds,batch_size=BATCH_SIZE,
                                           shuffle=shuffle,num_workers=2,
                                           pin_memory=True)
    return loader(train_set,True), loader(val_set,False), loader(test_set,False)


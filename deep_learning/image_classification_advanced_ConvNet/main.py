#!/usr/bin/env python3
import math, time, pickle
from datetime import datetime
from pathlib import Path

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingWarmRestarts, LinearLR, StepLR, LambdaLR

from utils import get_lr, plot_history, loaders, init_weights_He_normal, init_weights_He_uniform, custom_lr_schedule, set_seed
from config import EXPERIMENTS_VGG, EXPERIMENTS_RES, EXPERIMENTS_SE_RES, EXPERIMENTS_CBAM_RES


# -------------------------- Global Parameters --------------------------
SEED               = 2
RUNS_ROOT          = Path("./runs_pytorch")
DATA_ROOT          = Path("./data")

# VGG params
PATCH_F            = 2
PATCH_NF           = 64
VGG_BLOCKS         = 3
N_RES              = 3
FC_DIM             = 128
USE_BN             = True
DROPOUT            = 0.2
DROPOUT_VAR        = False
GLOBAL_AVG_POOL    = False
REMOVE_LAST_MAXPOOL= False
CONV_DOWN_SAMP     = False

# Res params
SHORTCUT           = 'Identity'
SE_ACTIVATE        = False
CBAM_ACTIVATE      = False
REDUCTION          = 8
MIN_DIM_SE         = 4

# Common params
MODEL              = 'VGG'
DATASET            = 'CIFAR10'
LR                 = 3e-4
WEIGHT_DECAY       = 5e-4
ETA_MIN, ETA_MAX   = LR/10, LR
STEP0              = 400
STEP_SIZE          = 6000
MOMENTUM           = 0.9
WARM_END           = 500
START_FACTOR       = 0.1
N_CYCLES           = 10
BATCH_SIZE         = 128
EPOCHS             = 10
LABEL_SMOOTH       = 0.1
AUGMENT            = True
SCHEDULER          = "CyclicLR"
OPTIMIZER          = "Adam"

EVAL_EVERY         = STEP0 // 2

EXP_ID             = "default"   # filled automatically

torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------- VGG-based Network --------------------------
# Patchify
class PatchifyStem(nn.Module):
    def __init__(self, f, nf):
        super().__init__()
        self.conv = nn.Conv2d(3, nf, f, stride=f)
    def forward(self, x): return F.relu(self.conv(x))

# VGG block
class VGGBlock(nn.Module):
    def __init__(self, in_c, out_c, use_bn=False, pool=True, p=0):
        super().__init__()
        layers = []
        for i in range(2):
            if i == 1 and CONV_DOWN_SAMP:
                layers.append(nn.Conv2d(in_c, out_c, 3, stride=2, padding=1,
                                        bias=not use_bn))
            else:
                layers.append(nn.Conv2d(in_c, out_c, 3, padding=1,
                                    bias=not use_bn))
            if use_bn: layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c
        if pool : layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

        self.drop = nn.Dropout(p)
        
    def forward(self, x): return self.drop(self.block(x))
    

# Vanilla VGG-based Network
class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = PatchifyStem(PATCH_F, PATCH_NF)

        blocks = []
        in_c, out_c = PATCH_NF, PATCH_NF
        for b in range(VGG_BLOCKS):
            if b: out_c *= 2
            pool_flag = not ((REMOVE_LAST_MAXPOOL and b == VGG_BLOCKS-1) or CONV_DOWN_SAMP)
            if DROPOUT_VAR:
                blocks.append(VGGBlock(in_c, out_c, USE_BN, pool_flag, DROPOUT + 0.1*b))
            else:
                blocks.append(VGGBlock(in_c, out_c, USE_BN, pool_flag, DROPOUT))
            in_c = out_c
        self.vgg = nn.Sequential(*blocks)

        self.drop = nn.Dropout(DROPOUT + 0.1*VGG_BLOCKS) if DROPOUT_VAR else nn.Dropout(DROPOUT)

        if GLOBAL_AVG_POOL:
            self.head = nn.Sequential(
                nn.Conv2d(in_c, 10, 1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        else:
            pools = VGG_BLOCKS - (1 if REMOVE_LAST_MAXPOOL else 0)
            feature_dim = 1024 if CONV_DOWN_SAMP else in_c * (32//PATCH_F // (2**pools))**2
            self.fc1 = nn.Linear(feature_dim, FC_DIM)
            self.fc2 = nn.Linear(FC_DIM, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.vgg(x)
        if GLOBAL_AVG_POOL:
            return self.head(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)
    
    
# -------------------------- Res Network --------------------------
# Res Block
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c

        # Res Block
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )

        if CBAM_ACTIVATE:
            self.att = CBAMBlock(out_c, REDUCTION)
        elif SE_ACTIVATE:
            self.att = SEBlock(out_c, REDUCTION)
        else:
            self.att = None


        # Shortcut if needed
        if SHORTCUT == 'Projection' and (stride != 1 or in_c != out_c):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        elif SHORTCUT == 'Identity' and (stride != 1 or in_c != out_c):
            self.shortcut = 'id'
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.att:
            out = self.att(out)

        # Shortcut if needed
        if self.shortcut == 'id': # identity shortcut
            identity = identity[:, :, ::self.stride, ::self.stride]
            pad_c = self.out_c - self.in_c
            identity = F.pad(identity, (0, 0, 0, 0, 0, pad_c), mode='constant', value=0)
        else: # projection shortcut or no shortcut
            identity = self.shortcut(identity)

        out += identity
        return self.relu(out)

# SE Block
class SEBlock(nn.Module):
    def __init__(self, in_c, r):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        squeeze_dim = max(MIN_DIM_SE, in_c // r) # avoid over-squeezing
        self.fc = nn.Sequential(
            nn.Linear(in_c, squeeze_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_dim, in_c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
# Channel Attention Block
class ChannelAttention(nn.Module):
    def __init__(self, in_c, reduction=16):
        super().__init__()
        squeeze_dim = max(MIN_DIM_SE, in_c // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_c, squeeze_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_dim, in_c, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = torch.mean(x, dim=[2, 3], keepdim=True)
        max_pool = torch.max(x, dim=2, keepdim=True)[0]
        max_pool = torch.max(max_pool, dim=3, keepdim=True)[0]
        
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out

# Spatial Attention Block
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

# CBAM Block
class CBAMBlock(nn.Module):
    def __init__(self, in_c, reduction=8):
        super().__init__()
        self.channel_att = SEBlock(in_c, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        sa = self.spatial_att(x)
        return x * sa

# Res Network
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Res Blocks
        self.in_c = 16
        blocks = []
        channels = [16, 32, 64]
        for i, out_c in enumerate(channels):
            for b in range(N_RES):
                if (b == 0 and i > 0): # check if shortcut is needed
                    stride = 2
                else:
                    stride = 1
                blocks.append(ResBlock(self.in_c, out_c, stride=stride))
                self.in_c = out_c
        self.res = nn.Sequential(*blocks)

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Fully-Connected Layer
        if DATASET == 'CIFAR100':
            self.fc = nn.Linear(self.in_c, 100)
        else:
            self.fc = nn.Linear(self.in_c, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.res(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -------------------------- Training --------------------------
# Training cycle
def train():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model type and initialization
    if MODEL == "VGG":
        model = VGGNet()
        model.apply(init_weights_He_uniform)
    elif MODEL == "Res":
        model = ResNet()
        model.apply(init_weights_He_normal)
    model.to(device)
        
    # Split parameters before applying weight decay
    decay, no_decay = [], []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if isinstance(module, nn.BatchNorm2d) or name.endswith('bias'):
                no_decay.append(param)
            else:
                decay.append(param)
        
    # Criterion and Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    if OPTIMIZER == "Adam":
        optimizer = optim.AdamW([{'params': no_decay, 'weight_decay': 0.0}, 
                                 {'params': decay, 'weight_decay': WEIGHT_DECAY}], lr=LR)
    elif OPTIMIZER == "SGD":
        optimizer = optim.SGD([{'params': no_decay, 'weight_decay': 0.0}, 
                               {'params': decay, 'weight_decay': WEIGHT_DECAY}],
                               lr=LR, momentum=MOMENTUM)
                                   
    # Scheduler
    if SCHEDULER == "Decay":
        sched = LambdaLR(optimizer, lr_lambda=custom_lr_schedule) # custom decay
    elif SCHEDULER == "CyclicLR":
        sched = CyclicLR(optimizer, base_lr=ETA_MIN, max_lr=ETA_MAX,
                         step_size_up=STEP0, cycle_momentum=False,
                         mode='triangular')
    elif SCHEDULER == "StepLR":
        sched = StepLR(optimizer, step_size=STEP_SIZE, gamma=0.1)
    elif SCHEDULER == "CosineAnnealWarmRest":
        sched = CosineAnnealingWarmRestarts(optimizer, STEP0*2, T_mult=2,
                                            eta_min=ETA_MIN)
    else:
        raise ValueError("unknown scheduler")

    # Load data
    train_loader, val_loader, test_loader = loaders(DATASET, AUGMENT, DATA_ROOT, SEED, BATCH_SIZE)

    hist = {k:[] for k in ('step','train_loss','val_loss',
                           'train_acc','val_acc','eta')}
    step = 0

    run_dir = RUNS_ROOT / f"{EXP_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logf = open(run_dir/'metrics.csv','w',buffering=1)
    logf.write("step,eta,train_loss,val_loss,train_acc,val_acc\n")

    # Training loop
    best_val = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for imgs,labels in train_loader:
            imgs,labels = imgs.to(device),labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            sched.step()

            if step % EVAL_EVERY == 0:
                acc = (logits.argmax(1)==labels).float().mean().item()
                v_loss,v_acc = evaluate(model,val_loader,criterion,device)
                lr_now = get_lr(optimizer)
                hist['step'].append(step)
                hist['train_loss'].append(loss.item())
                hist['val_loss'].append(v_loss)
                hist['train_acc'].append(acc)
                hist['val_acc'].append(v_acc)
                hist['eta'].append(lr_now)
                logf.write(f"{step},{lr_now:.4e},{loss.item():.4e},"
                           f"{v_loss:.4e},{acc:.4f},{v_acc:.4f}\n")
                print(f"[{step:06d}] loss {loss.item():.3f} ({v_loss:.3f})  "
                      f"acc {acc*100:5.2f}% ({v_acc*100:5.2f}%)  "
                      f"η={lr_now:.2e}")
                if v_acc > best_val:
                    best_val = v_acc
                    torch.save(model.state_dict(), run_dir/'best.pth')
            step += 1

    model.load_state_dict(torch.load(run_dir/'best.pth'))
    test_loss,test_acc = evaluate(model,test_loader,criterion,device)
    train_loss,train_acc = evaluate(model,train_loader,criterion,device)
    summary = {
    "final_train_acc": round(train_acc * 100, 2),
    "final_test_acc": round(test_acc * 100, 2),
    "best_val_acc": round(best_val * 100, 2)
    }
    with open(run_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"{EXP_ID}: best val {best_val*100:.2f}% → test {test_acc*100:.2f}%")
    plot_history(hist, run_dir/'training_curves.png')
    with open(run_dir/'history.pkl','wb') as f: pickle.dump(hist,f)

def evaluate(model, loader, criterion, device):
    was_training = model.training # store mode
    model.eval(); loss_sum=acc_sum=n=0
    with torch.no_grad():
        for imgs,labels in loader:
            imgs,labels=imgs.to(device),labels.to(device)
            logits=model(imgs)
            loss=criterion(logits,labels)
            b=labels.size(0)
            loss_sum+=loss.item()*b
            acc_sum +=(logits.argmax(1)==labels).float().sum().item()
            n+=b
    if was_training:
        model.train() # set back training mode
    return loss_sum/n, acc_sum/n


# -------------------------- Experiment Configurations --------------------------
def apply_cfg(cfg):
    if cfg['MODEL'] == 'Res': # update Res-related parameters
        globals().update({
            'MODEL':              cfg['MODEL'],
            'SHORTCUT':           cfg['SHORTCUT'],
            'SE_ACTIVATE':        cfg['SE_ACTIVATE'],
            'CBAM_ACTIVATE':      cfg['CBAM_ACTIVATE'],
            'LR':                 cfg['LR'],
            'N_RES':              cfg['N_RES'],
            'WEIGHT_DECAY':       cfg['WEIGHT_DECAY'],
            'AUGMENT':            cfg['AUGMENT'],
            'LABEL_SMOOTH':       cfg['LABEL_SMOOTH'],
            'SCHEDULER':          cfg['SCHEDULER'],
            'EPOCHS':             cfg['EPOCHS'],
            'OPTIMIZER':          cfg['OPTIMIZER'],
            'EXP_ID':             cfg['id'],
            'DATASET':            cfg['DATASET']
        })
    elif cfg['MODEL'] == 'VGG': # update VGG-related parameters
        globals().update({
            'MODEL':              cfg['MODEL'],
            'LR':                 cfg['LR'],
            'PATCH_F':            cfg['PATCH_F'],
            'PATCH_NF':           cfg['PATCH_NF'],
            'VGG_BLOCKS':         cfg['VGG_BLOCKS'],
            'FC_DIM':             cfg['FC_DIM'],
            'LABEL_SMOOTH':       cfg['LABEL_SMOOTH'],
            'GLOBAL_AVG_POOL':    cfg['GLOBAL_AVG_POOL'],
            'DROPOUT':            cfg['DROPOUT'],
            'DROPOUT_VAR':        cfg['DROPOUT_VAR'],
            'USE_BN':             cfg['USE_BN'],
            'REMOVE_LAST_MAXPOOL':cfg['REMOVE_LAST_MAXPOOL'],
            'CONV_DOWN_SAMP':     cfg['CONV_DOWN_SAMP'],
            'WEIGHT_DECAY':       cfg['WEIGHT_DECAY'],
            'AUGMENT':            cfg['AUGMENT'],
            'SCHEDULER':          cfg['SCHEDULER'],
            'EPOCHS':             cfg['EPOCHS'],
            'OPTIMIZER':          cfg['OPTIMIZER'],
            'EXP_ID':             cfg['id'],
        })
    globals()['ETA_MIN'], globals()['ETA_MAX'] = cfg['LR']/10, cfg['LR']


# -------------------------- Main --------------------------
if __name__=="__main__":
    RUNS_ROOT.mkdir(exist_ok=True)
    
    # VGG configurations
    for cfg in EXPERIMENTS_VGG:
        apply_cfg(cfg)
        print(cfg)
        train()      
    
    # Res configurations
    for cfg in EXPERIMENTS_RES:
        apply_cfg(cfg)
        print(cfg)
        train()
    
    # SE-Res configurations
    for cfg in EXPERIMENTS_SE_RES:
        apply_cfg(cfg)
        print(cfg)
        train()
        
    # CBAM-Res configurations
    for cfg in EXPERIMENTS_CBAM_RES:
        apply_cfg(cfg)
        print(cfg)
        train()
   
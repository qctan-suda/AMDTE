import argparse
import os
import csv
from datetime import datetime

import torch

from dataloader.dataset_combine_imf1 import AudioDataset
from torch.utils.data import DataLoader
from model.model_combine_imf1 import TFMamba
from torch import optim
from tqdm import tqdm
import numpy as np
from functools import partial
import math


def main():
    parser = argparse.ArgumentParser(description="Position estimation based on Audio used by Mamba")

    parser.add_argument("--audio_path", type=str, default="datasets/MMAUD/Data-M/combine_audio_npy/", help="load data from the path")
    parser.add_argument("--image_path", type=str, default="datasets/MMAUD/Data-M/image/", help="load image data from the path")
    parser.add_argument("--gt_cls_path", type=str, default="datasets/MMAUD/Data-M/label/", help="the gt type of uav")
    parser.add_argument("--gt_position_path", type=str, default="datasets/MMAUD/Data-M/gt/", help="the gt 3d position of uav")

    parser.add_argument("--save_path", type=str, default="output_combine_imf1/output_new_36_1/", help="the path to save model")

    parser.add_argument("--train_split_path", type=str, default="datasets/MMAUD/Data-M/annotation/annotation_train_all/trainval.txt",
                        help="the file of train, format: cls/file_name.npy, eg:0/111.npy")
    parser.add_argument("--val_split_path", type=str, default="datasets/MMAUD/Data-M/annotation/annotation_test_all/trainval.txt",
                        help="the file of train, format: cls/file_name.npy, eg:0/111.npy")

    parser.add_argument("--batch_size", type=int, default=512, help="data size of per batch")
    parser.add_argument("--train_epoch", type=int, default=200, help="number of training epochs")
    parser.add_argument("--workers", type=int, default=12, help="number of workers for dataloader")
    parser.add_argument("--gpu", type=str, default="cuda:1", help="training on the gpu device")
    parser.add_argument("--resume", type=str, default="output_combine_imf1/output_new_35/best_model_200_0.046099.pth", help="training on the gpu device")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    log_file = os.path.join(args.save_path, "training_log.csv")
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Train Cls Loss", "Train Pos Loss", "Val Loss", "Val Cls Loss", "Val Pos Loss", "LR", "Time"])

    train_dataset = AudioDataset(args.train_split_path, args.gt_cls_path, args.gt_position_path, args.audio_path)
    test_dataset = AudioDataset(args.val_split_path, args.gt_cls_path, args.gt_position_path, args.audio_path)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)

    model = TFMamba(num_cls=5, mode="train")

    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    lr_fun = lr_adjust_fun("decay", 0.001, 0.0001 * 1, args.train_epoch)

    pos_loss = torch.nn.L1Loss()
    cls_loss = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    for epoch in range(args.train_epoch):
        set_optim_lr(optimizer, lr_fun, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"current lr: {current_lr}")

        train_total_loss, train_cls_loss, train_pos_loss = train_mode(model, train_loader, optimizer, pos_loss, cls_loss, device)
        val_total_loss, val_cls_loss, val_pos_loss = valid_mode(model, test_loader, pos_loss, cls_loss, device)
        print(f"Epoch {epoch + 1}/{args.train_epoch}, "
              f"Train Loss: {train_total_loss:.6f}, "
              f"Train Cls Loss: {train_cls_loss:.6f}, "
              f"Train Pos Loss: {train_pos_loss:.6f}, "
              f"Val Loss: {val_total_loss:.6f}, "
              f"Val Cls Loss: {val_cls_loss:.6f}, "
              f"Val Pos Loss: {val_pos_loss:.6f}")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{train_total_loss:.6f}",
                f"{train_cls_loss:.6f}",
                f"{train_pos_loss:.6f}",
                f"{val_total_loss:.6f}",
                f"{val_cls_loss:.6f}",
                f"{val_pos_loss:.6f}",
                f"{current_lr:.8f}",
                current_time
            ])
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save(model.state_dict(), os.path.join(args.save_path, f"best_model_{epoch + 1}_{val_total_loss:.6f}.pth"))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, f"epoch{epoch + 1}_val_loss_{val_total_loss:.6f}.pth"))


def train_mode(model, train_loader, optimizer, pos_loss, cls_loss, device):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_pos_loss = 0.0
    
    for data in tqdm(train_loader, total=len(train_loader), unit="batch"):
        spectrogram, cls_gt, pos_gt = [d.to(device) for d in data]
        optimizer.zero_grad()
        cls_pred, pos_pred = model(spectrogram)

        loss_cls = cls_loss(cls_pred, cls_gt)
        loss_pos = pos_loss(pos_pred, pos_gt)
        loss = loss_cls + 2 * loss_pos
        
        loss.backward()
        optimizer.step()
        
        
        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_pos_loss += loss_pos.item()
    
    
    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_pos_loss = total_pos_loss / len(train_loader)
    return avg_loss, avg_cls_loss, avg_pos_loss


def set_optim_lr(optim, lr_adjust_fun, epoch):
    lr = lr_adjust_fun(epoch)
    for param_group in optim.param_groups:
        param_group["lr"] = lr


def lr_adjust_fun(lr_decay_type, lr, min_lr, total_iters,
                  warmup_iters_ratio=0.05,
                  warmup_lr_ratio=0.1,
                  no_aug_iter_ratio=0.05,
                  step_num=10):

    def warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi*(iters - warmup_total_iters) / (total_iters - warmup_total_iters -no_aug_iter)))
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        fun = partial(warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        fun = partial(step_lr, lr, decay_rate, step_size)
    return fun


def valid_mode(model, valid_loader, pos_loss, cls_loss, device):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_pos_loss = 0.0
    
    with torch.no_grad():
        for data in tqdm(valid_loader, total=len(valid_loader), unit="batch"):
            spectrogram, cls_gt, pos_gt = [d.to(device) for d in data]
            cls_pred, pos_pred = model(spectrogram)

            loss_cls = cls_loss(cls_pred, cls_gt)
            loss_pos = pos_loss(pos_pred, pos_gt)
            loss = loss_cls + loss_pos
            
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_pos_loss += loss_pos.item()
    
    avg_loss = total_loss / len(valid_loader)
    avg_cls_loss = total_cls_loss / len(valid_loader)
    avg_pos_loss = total_pos_loss / len(valid_loader)
    return avg_loss, avg_cls_loss, avg_pos_loss


if __name__ == "__main__":
    main()
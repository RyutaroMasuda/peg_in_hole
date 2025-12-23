#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import os
import sys
import numpy as np
import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
sys.path.append("../eipl/")
from eipl.model import SARNN
from eipl.data import MultimodalDataset
from eipl.utils import EarlyStopping, check_args, set_logdir, normalization, resize_img
import ipdb

# load own library
sys.path.append("./libs/")
from fullBPTT import fullBPTTtrainer#学習用自作コード

# argument parser
parser = argparse.ArgumentParser(
    description="Learning spatial autoencoder with recurrent neural network"
)
parser.add_argument("train_data_dir", type=str)
parser.add_argument("--model", type=str, default="sarnn")
parser.add_argument("--epoch", type=int, default=30000)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--rec_dim", type=int, default=50)
parser.add_argument("--k_dim", type=int, default=6)
parser.add_argument("--img_loss", type=float, default=0.5)
parser.add_argument("--joint_loss", type=float, default=1.0)
parser.add_argument("--pt_loss", type=float, default=0.1)
parser.add_argument("--heatmap_size", type=float, default=0.1)
parser.add_argument("--temperature", type=float, default=1e-4)
parser.add_argument("--stdev", type=float, default=0.02)
parser.add_argument("--lr", type=float, default=1e-3)# 1e-3
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--log_dir", default="log/")
parser.add_argument("--vmin", type=float, default=0.0)
parser.add_argument("--vmax", type=float, default=1.0)
parser.add_argument("--device", type=int, default=2)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--tag", help="Tag name for snap/log sub directory")
args = parser.parse_args()

# check args
args = check_args(args)

# calculate the noise level (variance) from the normalized range
stdev = args.stdev * (args.vmax - args.vmin)

# set device id
if args.device >= 0:
    device = "cuda:{}".format(args.device)
else:
    device = "cpu"

# load dataset
right_images_raw = np.load(f"{args.train_data_dir}/train/right_imgs.npy")
right_joints_raw = np.load(f"{args.train_data_dir}/train/teleop_right_arm_states.npy")
right_joint_bounds = np.load(f"{args.train_data_dir}/data/teleop_right_joint_bounds.npy")
# right_tactile_images = np.load(f"{args.train_data_dir}/train/right_digit_imgs.npy")

# normalize data
minmax = [args.vmin, args.vmax]

right_images = normalization(right_images_raw.transpose(0,1,4,2,3), (0, 255), minmax)
right_joints = normalization(right_joints_raw, right_joint_bounds, minmax)
# right_tactile_images = normalization(right_tactile_images.transpose(0,1,4,2,3), (0, 255), minmax)

# add gaussian noize for getting robust data
train_dataset = MultimodalDataset(right_images, right_joints, device=device, stdev=stdev)

print("[DEBUG] creating train loader...")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=0
)

print("[DEBUG] loading test data...")
print("[DEBUG] test images_raw shape:", right_images.shape)
print("[DEBUG] test joints_raw shape:", right_joints.shape)

right_images_raw = np.load(f"{args.train_data_dir}/test/right_imgs.npy")
right_joints_raw = np.load(f"{args.train_data_dir}test/teleop_right_arm_states.npy")

right_images = normalization(right_images_raw.transpose(0,1,4,2,3), (0, 255), minmax)
right_joints = normalization(right_joints_raw, right_joint_bounds, minmax)

print("[DEBUG] test image shape after transpose:", right_images.shape)
print("[DEBUG] image min/max:", right_images.min(), right_images.max())

print("[DEBUG] creating test dataset...")
test_dataset = MultimodalDataset(right_images, right_joints, device=device, stdev=None)

print("[DEBUG] creating test loader...")
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=0
)

print("[DEBUG] defining model...")


# define model
model = SARNN(
    rec_dim=args.rec_dim,
    joint_dim=9,
    k_dim=args.k_dim,
    heatmap_size=args.heatmap_size,
    temperature=args.temperature,
    im_size=[64, 64],
)

# torch.compile makes PyTorch code run faster
if args.compile:
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)

# set optimizer
optimizer = optim.Adam(model.parameters(), eps=1e-07 , lr=args.lr)

# load trainer/tester class
loss_weights = [args.img_loss, args.joint_loss, args.pt_loss]
trainer = fullBPTTtrainer(model, optimizer, loss_weights=loss_weights, device=device)

### training main
log_dir_path = set_logdir("./" + args.log_dir, args.tag)
save_name = os.path.join(log_dir_path, "SARNN.pth")
writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)
early_stop = EarlyStopping(patience=1000)

# ipdb.set_trace()
with tqdm(range(args.epoch)) as pbar_epoch:
    for epoch in pbar_epoch:
        # train and test
        train_results = trainer.process_epoch(train_loader)
        with torch.no_grad():
            test_results = trainer.process_epoch(test_loader, training=False)
        
        # TensorBoardに記録
        writer.add_scalar("Loss/train_total_loss", train_results['total_loss'], epoch)
        writer.add_scalar("Loss/test_total_loss", test_results['total_loss'], epoch)
        writer.add_scalar("Loss/train_img_loss", train_results['img_loss'], epoch)
        writer.add_scalar("Loss/test_img_loss", test_results['img_loss'], epoch)
        writer.add_scalar("Loss/train_joint_loss", train_results['joint_loss'], epoch)
        writer.add_scalar("Loss/test_joint_loss", test_results['joint_loss'], epoch)
        writer.add_scalar("Loss/train_pt_loss", train_results['pt_loss'], epoch)
        writer.add_scalar("Loss/test_pt_loss", test_results['pt_loss'], epoch)
        # early stop
        save_ckpt, _ = early_stop(test_results['total_loss'])

        if save_ckpt:
            save_name = os.path.join(log_dir_path, f"SARNN{epoch}.pth")
            trainer.save(epoch, [train_results['total_loss'], test_results['total_loss']], save_name)
            print(f"保存されているepoch:{epoch}")

        # print process bar
        pbar_epoch.set_postfix(OrderedDict(
            train_loss=train_results['total_loss'], 
            test_loss=test_results['total_loss'],
            train_img_loss=train_results['img_loss'],
            train_joint_loss=train_results['joint_loss'],
            train_pt_loss=train_results['pt_loss'],

        ))
        pbar_epoch.update()
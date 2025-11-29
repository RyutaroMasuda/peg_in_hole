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
import cv2
import torch.optim as optim
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
sys.path.append("/home/masuda/work/eipl/")
from eipl.model import SARNN
from eipl.data import MultimodalDataset
from eipl.utils import EarlyStopping, check_args, set_logdir, normalization, resize_img
import ipdb
import matplotlib.pyplot as plt

# load own library
sys.path.append("./libs/")
from fullBPTT import fullBPTTtrainer

# argument parser
parser = argparse.ArgumentParser(
    description="Learning spatial autoencoder with recurrent neural network"
)
parser.add_argument("--model", type=str, default="sarnn")
parser.add_argument("--epoch", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--rec_dim", type=int, default=50)
parser.add_argument("--k_dim", type=int, default=5)
parser.add_argument("--img_loss", type=float, default=0.1)
parser.add_argument("--joint_loss", type=float, default=1.0)
parser.add_argument("--pt_loss", type=float, default=0.1)
parser.add_argument("--heatmap_size", type=float, default=0.1)
parser.add_argument("--temperature", type=float, default=1e-4)
parser.add_argument("--stdev", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--log_dir", default="log/")
parser.add_argument("--vmin", type=float, default=0.0)
parser.add_argument("--vmax", type=float, default=1.0)
parser.add_argument("--device", type=int, default=0)
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
minmax = [args.vmin, args.vmax]
print("[DEBUG] loading training images...")
images_raw = np.load("./data/20250515/train/right_imgs.npy")
dir_path="20250515"
os.makedirs(dir_path, exist_ok=True)

print(f"'{dir_path}' を作成しました")
for b in range(images_raw.shape[0]):
    frame_height,frame_width=images_raw.shape[2],images_raw.shape[3]
    output_video_path=f"./fig/video_{b}.mp4"

    #動画の設定 
    fps=10
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')#動画のエンコーディング形式
    
    #VideoWriterオブジェクトを作成
    out=cv2.VideoWriter(output_video_path,fourcc,fps,(frame_width,frame_height))

    #画像をフレームとして追加
    for i in range(images_raw.shape[1]):
        image=images_raw[b,i]
        image_resized=cv2.resize(image,(frame_width,frame_height))
        out.write(image_resized)
    out.release()
    print(f"動画が{output_video_path}に保存されました")


print("[DEBUG] loading training joints...")
joints_raw = np.load("./data/20250515/train/right_arm_states.npy")
for i in range(joints_raw.shape[0]):
        plt.figure()
        plt.plot(joints_raw[i])
        plt.savefig(f"./fig/sample_states{i}.png")
# ipdb.set_trace()

print("[DEBUG] loading joint bounds...")
joint_bounds = np.load("./data/20250515/param/arm_state_bound.npy")

print("[DEBUG] normalizing training data...")
images_raw = resize_img(images_raw, size=(64,64))
images = normalization(images_raw.transpose(0, 1, 4, 2, 3), (0, 255), minmax)
joints = normalization(joints_raw, joint_bounds, minmax)

print("[DEBUG] creating train dataset...")
train_dataset = MultimodalDataset(images, joints, device=device, stdev=stdev)

print("[DEBUG] creating train loader...")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=0
)

print("[DEBUG] loading test data...")
print("[DEBUG] test images_raw shape:", images_raw.shape)
print("[DEBUG] test joints_raw shape:", joints_raw.shape)

images_raw = np.load("./data/20250515/test/right_imgs.npy")
joints_raw = np.load("./data/20250515/test/right_arm_states.npy")
# ここから
# images = images_raw.transpose(0, 1, 4, 2, 3)
images_raw = resize_img(images_raw, size=(64,64))
images = normalization(images_raw.transpose(0, 1, 4, 2, 3), (0, 255), minmax)
joints = normalization(joints_raw, joint_bounds, minmax)

print("[DEBUG] test image shape after transpose:", images.shape)
print("[DEBUG] image min/max:", images.min(), images.max())

if np.isnan(images).any():
    print("[ERROR] NaN detected in test images")

joints = joints_raw
if np.isnan(joints).any():
    print("[ERROR] NaN detected in test joints")
# ここまで
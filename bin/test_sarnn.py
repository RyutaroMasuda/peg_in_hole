#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import os
import sys
import torch
import ipdb
import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
sys.path.append("../eipl")
from eipl.data import SampleDownloader, WeightDownloader
from eipl.model import SARNN
from eipl.utils import normalization
from eipl.utils import restore_args, tensor2numpy, deprocess_img, resize_img
from matplotlib.animation import PillowWriter 


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None)
parser.add_argument("--idx", type=str, default="0")
parser.add_argument("--input_param", type=float, default=1.0)
parser.add_argument("--pretrained", action="store_true")
args = parser.parse_args()

# check args
assert args.filename or args.pretrained, "Please set filename or pretrained"

# load pretrained weight
if args.pretrained:
    WeightDownloader("airec", "grasp_bottle")
    args.filename = os.path.join(
        os.path.expanduser("~"), ".eipl/airec/grasp_bottle/weights/SARNN/model.pth"
    )

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))
idx = int(args.idx)

# load dataset
minmax = [params["vmin"], params["vmax"]]
grasp_data = SampleDownloader("airec", "grasp_bottle", img_format="HWC")
_images=np.load("./data/20251112_redsylinder2/test/right_imgs.npy")
#_images=np.load("./data/20250710/test/right_imgs.npy")
_joints=np.load("./data/20251112_redsylinder2/test/right_arm_states.npy")
#_joints=np.load("./data/20250710/test/right_arm_states.npy")
# _images_tactile=np.load("./data/20251112_redsylinder2/test/right_digit_imgs.npy")
# _images = resize_img(_images, (64, 64)) #make_dataset.py時にすでにやってる気がする
# _images_tactile = resize_img(_images_tactile,(64,64))
images = _images[idx]
joints = _joints[idx]
# images_tactile = _images_tactile[idx]
# prepare joint_bounds correctly
joint_bounds_raw = np.load("./data/20251112_redsylinder2/test/teleop_right_arm_states.npy")
#joint_bounds_raw = np.load("./data/20250710/test/teleop_right_arm_states.npy")
joint_bounds_min = joint_bounds_raw.min(axis=(0, 1))
joint_bounds_max = joint_bounds_raw.max(axis=(0, 1))
joint_bounds = np.vstack([joint_bounds_min, joint_bounds_max])  # shape = (2, 9)

print(
    "images shape:{}, min={}, max={}".format(images.shape, images.min(), images.max())
)
print(
    "joints shape:{}, min={}, max={}".format(joints.shape, joints.min(), joints.max())
)
# print(
#     "tactile images shape:{}, min={}, max={}".format(images_tactile.shape, images_tactile.min(), images_tactile.max())
# )

# define model
model = SARNN(
    rec_dim=params["rec_dim"],
    joint_dim=9,
    k_dim=params["k_dim"],
    heatmap_size=params["heatmap_size"],
    temperature=params["temperature"],
    im_size=[64, 64], 
)

if params["compile"]:
    model = torch.compile(model)

# load weight
ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference
img_size = 64
image_list, joint_list = [], []
enc_pts_list, dec_pts_list = [], []
# enc_tactile_pts_list, dec_tactile_pts_list = [], []
state = None
nloop = len(images)
input_joint_list = []
for loop_ct in range(nloop):
    # load data and normalization
    img_t = images[loop_ct].transpose(2, 0, 1) # (H,W,C) to (C,H,W)
    img_t = torch.Tensor(np.expand_dims(img_t, 0)) #(C,H,W) to (B,C,H,W)
    img_t = normalization(img_t, (0, 255), minmax) 

    joint_t = torch.Tensor(np.expand_dims(joints[loop_ct], 0))
    joint_t = normalization(joint_t, joint_bounds, minmax)

    # img_tactile_t = images_tactile[loop_ct].transpose(2,0,1)
    # img_tactile_t = torch.Tensor(np.expand_dims(img_tactile_t,0))
    # img_tactile_t = normalization(img_tactile_t, (0,255),minmax)

    # import ipdb;ipdb.set_trace()
    # closed loop. y_imageは予測値でimg_tは実測値。過去の予測値をどれだけ混ぜるか。基本は混ぜなくて良い
    if loop_ct > 0:
        img_t = args.input_param * img_t + (1.0 - args.input_param) * y_image
        joint_t = args.input_param * joint_t + (1.0 - args.input_param) * y_joint
        # img_tactile_t = args.input_param * img_tactile_t + (1.0 - args.input_param) * y_image_tactile

    # predict rnn
    y_image, y_joint, enc_pts, dec_pts, state = model(img_t, joint_t, state)# すべて0,1正規化されて出力
    # denormalization
    pred_image = tensor2numpy(y_image[0])
    pred_image = deprocess_img(pred_image, params["vmin"], params["vmax"]) 
    pred_image = pred_image.transpose(1, 2, 0)
    pred_joint = tensor2numpy(y_joint[0])
    pred_joint = normalization(pred_joint, minmax, joint_bounds)
    # pred_image_tactile = tensor2numpy(y_image_tactile[0])
    # pred_image_tactile = deprocess_img(pred_image_tactile, params["vmin"], params["vmax"])
    # pred_image_tactile = pred_image_tactile.transpose(1,2,0)

    # send pred_joint to robot
    # send_command(pred_joint)
    # pub.publish(pred_joint)

    # append data
    image_list.append(pred_image)
    joint_list.append(pred_joint)
    # image_tactile_list.append(pred_image_tactile)
    enc_pts_list.append(tensor2numpy(enc_pts[0]))
    dec_pts_list.append(tensor2numpy(dec_pts[0]))
    # enc_tactile_pts_list.append(tensor2numpy(enc_tactile_pts[0]))
    # dec_tactile_pts_list.append(tensor2numpy(dec_tactile_pts[0]))
    # save joint_t
    input_joint_list.append(tensor2numpy(joint_t[0]))
    # ipdb.set_trace()

    print("loop_ct:{}, joint:{}".format(loop_ct, pred_joint))

pred_image = np.array(image_list)
pred_joint = np.array(joint_list)
# pred_image_tactile = np.array(image_tactile_list)
# ipdb.set_trace()
# ipdb.set_trace()
# split key points
enc_pts = np.array(enc_pts_list)
dec_pts = np.array(dec_pts_list)
enc_pts = enc_pts.reshape(-1, params["k_dim"], 2) * img_size
dec_pts = dec_pts.reshape(-1, params["k_dim"], 2) * img_size
enc_pts = np.clip(enc_pts, 0, img_size)
dec_pts = np.clip(dec_pts, 0, img_size)
# ipdb.set_trace()
# split tactile keypoints
# enc_tactile_pts = np.array(enc_tactile_pts_list)
# dec_tactile_pts = np.array(dec_tactile_pts_list)
# enc_tactile_pts = enc_tactile_pts.reshape(-1, params["k_dim_tactile"], 2) * img_size
# dec_tactile_pts = dec_tactile_pts.reshape(-1, params["k_dim_tactile"], 2) * img_size
# enc_tactile_pts = np.clip(enc_tactile_pts, 0, img_size)
# dec_tactile_pts = np.clip(dec_tactile_pts, 0, img_size)

# convert input_joint_list to numpy array
input_joint_arr = np.array(input_joint_list)

print("joints shape:", joints.shape, "min:", np.nanmin(joints), "max:", np.nanmax(joints))
print("pred_joint shape:", pred_joint.shape, "min:", np.nanmin(pred_joint), "max:", np.nanmax(pred_joint))
print("input_joint_arr shape:", input_joint_arr.shape, "min:", np.nanmin(input_joint_arr), "max:", np.nanmax(input_joint_arr))

if np.any(np.isnan(joints)) or np.any(np.isnan(pred_joint)) or np.any(np.isnan(input_joint_arr)):
    print("Warning: NaN detected in data!")
if np.any(np.isinf(joints)) or np.any(np.isinf(pred_joint)) or np.any(np.isinf(input_joint_arr)):
    print("Warning: inf detected in data!")

# --- ループ後に追加 ---
def denormalize(normed, bounds):
    return normed * (bounds[1] - bounds[0]) + bounds[0]
input_joint_arr_denorm = denormalize(input_joint_arr, joint_bounds)

# plot images
T = len(images)
fig, ax = plt.subplots(2, 3, figsize=(18, 10), dpi=80)


def anim_update(i):
    # Clear all subplots
    for row in range(2):
        for col in range(3):
            ax[row, col].cla()

    # Top row: Images
    # plot camera image
    ax[0, 0].imshow(images[i, :, :, ::-1])
    for j in range(params["k_dim"]):
        ax[0, 0].plot(enc_pts[i, j, 0], enc_pts[i, j, 1], "bo", markersize=6)  # encoder
        ax[0, 0].plot(
            dec_pts[i, j, 0], dec_pts[i, j, 1], "rx", markersize=6, markeredgewidth=2
        )  # decoder
    ax[0, 0].axis("off")
    ax[0, 0].set_title("Input image")

    # plot predicted image
    ax[0, 1].imshow(pred_image[i, :, :, ::-1])
    ax[0, 1].axis("off")
    ax[0, 1].set_title("Predicted image")

    # plot joint angle overview
    ax[0, 2].set_ylim(
        min(np.nanmin(joints), np.nanmin(pred_joint), np.nanmin(input_joint_arr_denorm)) - 0.1,
        max(np.nanmax(joints), np.nanmax(pred_joint), np.nanmax(input_joint_arr_denorm)) + 0.1
    )
    ax[0, 2].set_xlim(0, T)
    n_joints = joints.shape[1] if joints.ndim == 2 else 1
    import matplotlib.cm as cm
    colors = [cm.get_cmap('tab10')(i) for i in range(n_joints)]
# 凡例を一度に表示するための空のリストを準備
    gt_lines = []
    pred_lines = []
    input_lines = []

    for joint_idx in range(n_joints):
        # 0からn_jointsまでのインデックスを色指定に利用
        color = colors[joint_idx]
        joint_label = f'Joint {joint_idx + 1}'

        # GT (実測値) - 破線
        gt_line, = ax[0, 2].plot(joints[:, joint_idx], linestyle="dashed", c=color, alpha=0.5)
        
        # Pred (予測値) - 実線
        pred_line, = ax[0, 2].plot(pred_joint[:i+1, joint_idx], c=color, alpha=1.0)
        
        # Input (入力値) - 点線や別のスタイルで区別
        input_line, = ax[0, 2].plot(input_joint_arr_denorm[:i+1, joint_idx], linestyle="dotted", c=color, alpha=0.5)
        
        # 凡例用のリストに色とジョイントラベルを紐づけて追加
        # ここでは、凡例はGT/Pred/Inputの3種類と、各ジョイントの色分けを両立させたいので、
        # 以下の方法でジョイントごとの凡例を表示します。

        # 各ジョイントの凡例ラインを一時的に格納
        gt_lines.append(gt_line)
        pred_lines.append(pred_line)
        input_lines.append(input_line)

    ax[0, 2].set_xlabel("Step")
    ax[0, 2].set_title("Joint angles overview")
    
    # 各ジョイントの色分け凡例を作成
    # 凡例ハンドルとラベルを作成
    joint_legend_handles = [gt_lines[i] for i in range(n_joints)]
    joint_legend_labels = [f'Joint {i + 1}' for i in range(n_joints)]
    
    # スタイルを示す凡例を作成（全てのジョイントで共通）
    # 最初のジョイントのラインスタイルを使用し、透明度を調整して凡例表示用のダミーを作成
    style_legend_handles = [
        ax[0, 2].plot([], [], linestyle="dashed", c="k", alpha=0.5, label='GT')[0],
        ax[0, 2].plot([], [], linestyle="-", c="k", alpha=1.0, label='Pred')[0],
        ax[0, 2].plot([], [], linestyle="dotted", c="k", alpha=0.5, label='Input')[0]
    ]
    style_legend_labels = ['GT', 'Pred', 'Input']

    # 1. スタイルの凡例（GT/Pred/Input）を表示
    style_legend = ax[0, 2].legend(style_legend_handles, style_legend_labels, loc='upper left', fontsize=8)
    
    # 2. ジョイントごとの色の凡例を表示（1.とは別の凡例として重ねて表示）
    # style_legend のアーティストを再利用して、2つ目の凡例に含めないようにする
    ax[0, 2].add_artist(style_legend)
    
    # loc='upper right' など、重ならない適切な位置を指定
    ax[0, 2].legend(joint_legend_handles, joint_legend_labels, loc='lower center', fontsize=8, ncol=2)
   

    # ax[1, 0].imshow(images_tactile[i, :, :, ::-1])
    # for j in range(params["k_dim_tactile"]):
    #     ax[1, 0].plot(enc_tactile_pts[i, j, 0], enc_tactile_pts[i, j, 1], "bo", markersize=6)  # encoder
    #     ax[1, 0].plot(
    #         dec_tactile_pts[i, j, 0], dec_tactile_pts[i, j, 1], "rx", markersize=6, markeredgewidth=2
    #     )  # decoder
    # ax[1, 0].axis("off")
    # ax[1, 0].set_title("Input image")

    # Add time step indicator
    fig.suptitle(f'Time Step: {i+1}/{T} (input_param={args.input_param})', fontsize=14)


ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
ani.save("./output/SARNN_{}_{}_{}.gif".format(params["tag"], idx, args.input_param),
         writer=PillowWriter(fps=10))

# If an error occurs in generating the gif animation or mp4, change the writer (imagemagick/ffmpeg).
# ani.save("./output/PCA_SARNN_{}.gif".format(params["tag"]), writer="imagemagick")
# ani.save("./output/PCA_SARNN_{}.mp4".format(params["tag"]), writer="ffmpeg")

print(f"Synchronized GIF saved to ./output/SARNN_synchronized_replay_{params['tag']}_{idx}_{args.input_param}.gif")

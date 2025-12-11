
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from natsort import natsorted
import os
import sys
from eipl.utils import normalization
import matplotlib.animation as anim
sys.path.append("/home/masuda/work/eipl")

parser = argparse.ArgumentParser()
parser.add_argument("npy_dir", type=str)
parser.add_argument("--freq", type=float, default=10)
parser.add_argument("--idx",type=int, default = 0)
args = parser.parse_args()

idx = int(args.idx)

files = glob.glob(os.path.join(args.npy_dir, "*.npy"))
files = natsorted(files)

topics_imgs = ["right_imgs", "left_imgs", "right_digit_imgs","right_hand_imgs", "left_hand_imgs"]
topics_joint = ["upperbody_left_arm_states","upperbody_right_arm_states"]
topics = ["right_imgs.npy", "left_imgs.npy", "right_digit_imgs.npy", "right_hand_imgs.npy", "left_hand_imgs.npy","left_arm_states.npy","right_arm_states.npy"]

fig,ax = plt.subplots(2,4,figsize=(28,10),dpi=120)
ax = ax.flatten()
# plt.subplots_adjust(wspace=0.4)
def anim_update(i,data_dict):
    for index in [0,1,2,3]:
        ax[index].cla()
    """左目と右目画像の描画"""
    ax[0].imshow(data_dict["left_imgs.npy"][idx,i,:,:,::-1])
    ax[0].axis("off")
    ax[0].set_title(f"Left eye:{i}")
    ax[1].imshow(data_dict["right_imgs.npy"][idx,i,:,:,::-1])
    ax[1].axis("off")
    ax[1].set_title(f"Right eye:{i}")

    """右手触覚画像の描画"""
    ax[6].imshow(data_dict["right_digit_imgs.npy"][idx,i,:,:,2])
    ax[6].axis("off")
    ax[6].set_title(f"Right tactile:{i}")

    """左手と右手画像の描画"""
    ax[4].imshow(data_dict["right_digit_imgs.npy"][idx,i,:,:,0])
    ax[4].axis("off")
    ax[4].set_title(f"Right tactile:{i}")
    ax[5].imshow(data_dict["right_digit_imgs.npy"][idx,i,:,:,1])
    ax[5].axis("off")
    ax[5].set_title(f"Right tactile:{i}")

    """左手関節データの描画"""
    left_joint=data_dict["left_arm_states.npy"][idx]
    left_joint_norm=normalization(left_joint,[left_joint.min(),left_joint.max()],[0.0,1.0])
    ax[3].set_ylim(0.0,1.0)
    ax[3].set_xlim(0.0,seq)
    # ax[6].plot()
    ax[3].plot(left_joint_norm,linestyle="dashed")
    for joint_idx in range(9):
        ax[3].plot(np.arange(i+1), left_joint_norm[:i+1,joint_idx])
    ax[3].set_xlabel("Step")
    ax[3].set_title("Left Joint angles")
    
    """右手関節データの描画"""
    right_joint=data_dict["right_arm_states.npy"][idx]
    right_joint_norm=normalization(right_joint,[right_joint.min(),right_joint.max()],[0.0,1.0])
    ax[2].set_ylim(0.0,1.0)
    ax[2].set_xlim(0.0,seq)
    # ax[6].plot()
    ax[2].plot(right_joint_norm,linestyle="dashed")
    for joint_idx in range(9):
        ax[2].plot(np.arange(i+1), right_joint_norm[:i+1,joint_idx])
    ax[2].set_xlabel("Step")
    ax[2].set_title("Right Joint angles")

    return ax
data_dict = {}
for file in files:
    key = os.path.basename(file)
    data_dict[key] = np.load(file)

seq = min([
    data_dict[key].shape[1] for key in topics  
])
# animationの作成. FuncAnimationにanim_update関数とframe数を入れたら作成可能.
ani = anim.FuncAnimation(fig,anim_update,frames=seq,interval=int(10000/10),fargs=(data_dict,))
base_name=os.path.basename(file).split(".")[0]    
folder_path=args.npy_dir.split(f"{base_name}")[0]
folder_name=folder_path.split("/")[1]
save_dir= f"./fig/{folder_name}/"
os.makedirs(save_dir,exist_ok=True)
save_path=os.path.join(save_dir,f"train_data_{idx}.gif")

ani.save(save_path, fps=args.freq)

print(f"GIF画像を保存しました:{save_path}")
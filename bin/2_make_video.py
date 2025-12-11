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
parser.add_argument("npz_dir", type=str)
parser.add_argument("--freq", type=float, default=10)
parser.add_argument("--collect_mode",type=str,default="pos5")
args = parser.parse_args()

if args.collect_mode == "plain":
    files = glob.glob(os.path.join(args.npz_dir, "*.bag"))
    files.sort()
elif args.collect_mode == "pos5":
    files = glob.glob(os.path.join(f"{args.npz_dir}/pos*/*.npz"))
    # import ipdb;ipdb.set_trace()


topics_imgs = ["right_imgs", "left_imgs", "right_digit_imgs","right_hand_imgs", "left_hand_imgs"]
topics_joint = ["upperbody_left_arm_states","upperbody_right_arm_states"]
topics = ["right_imgs", "left_imgs", "right_digit_imgs", "right_hand_imgs", "left_hand_imgs","upperbody_left_arm_states","upperbody_right_arm_states"]

fig,ax = plt.subplots(2,4,figsize=(28,10),dpi=120)
ax = ax.flatten()
# plt.subplots_adjust(wspace=0.4)
def anim_update(i,data_dict):
    for a in ax:
        a.cla()
    """左目と右目画像の描画"""
    ax[0].imshow(data_dict["left_imgs"][i][:,:,::-1])
    ax[0].axis("off")
    ax[0].set_title(f"Left eye:{i}")
    ax[1].imshow(data_dict["right_imgs"][i][:,:,::-1])
    ax[1].axis("off")
    ax[1].set_title(f"Right eye:{i}")
    """右手触覚画像の描画"""
    ax[6].imshow(data_dict["right_digit_imgs"][i][:,:,::-1])
    ax[6].axis("off")
    ax[6].set_title(f"Right tactile:{i}")
    """左手と右手画像の描画"""
    ax[4].imshow(data_dict["left_hand_imgs"][i][:,:,::-1])
    ax[4].axis("off")
    ax[4].set_title(f"Left hand:{i}")
    ax[5].imshow(data_dict["right_hand_imgs"][i][:,:,::-1])
    ax[5].axis("off")
    ax[5].set_title(f"Right eye:{i}")
    """左手関節データの描画"""
    left_joint=data_dict["upperbody_left_arm_states"]
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
    right_joint=data_dict["upperbody_right_arm_states"]
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

for file in files:
    data = np.load(file)
    # 存在するtopic名のリストを抽出
    available_topics_imgs = [t for t in topics_imgs if t in data.keys()]
    available_topics_joint = [t for t in topics_joint if t in data.keys()]
    print(f"データに含まれている画像topicは{available_topics_imgs}です")
    print(f"データに含まれている関節topicは{available_topics_joint}です")
    # 存在するtopicの辞書データを抽出
    data_dict = {}
    for t in available_topics_imgs + available_topics_joint:
        data_dict[t] = data[t]

    seq = min([
      data_dict[t].shape[0] for t in topics  
    ])
    # animationの作成. FuncAnimationにanim_update関数とframe数を入れたら作成可能.
    ani = anim.FuncAnimation(fig,anim_update,frames=seq,interval=int(10000/10),fargs=(data_dict,))
    if args.collect_mode == "plain":
        base_name=os.path.basename(file).split(".")[0]    
        folder_path=args.npz_dir.split(f"{base_name}")[0]
        folder_name=folder_path.split("/")[1]
        save_dir= f"./fig/{folder_name}/"
        os.makedirs(save_dir,exist_ok=True)
        save_path=os.path.join(save_dir,f"{base_name}.gif")
    elif args.collect_mode == "pos5":
        base = os.path.splitext(os.path.basename(file))[0] #.npyを除いたディレクトリ
        npy_root = os.path.join(args.npz_dir) # posi for i in range 5のあるディレクトリ
        rel_path = os.path.relpath(file, npy_root) # fileからみた.npyの相対ディレクトリ
        folder_path = npy_root.split("/")[1]
        save_dir = os.path.join("./fig", folder_path, os.path.dirname(rel_path))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, base + ".gif")

    ani.save(save_path, fps=args.freq)
    print(f"GIF画像を保存しました:{save_path}")
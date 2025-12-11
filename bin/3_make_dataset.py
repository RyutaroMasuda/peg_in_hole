import os
import sys
import cv2
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt
sys.path.append("../eipl")
from eipl.utils import resize_img, calc_minmax,list_to_numpy,get_mean_minmax,get_bounds
import random



parser = argparse.ArgumentParser()
parser.add_argument("input_dir",type=str)
parser.add_argument("--output_num",type=str,default="")
parser.add_argument("--collect_mode",type=str,default="pos5")
args=parser.parse_args()
if args.collect_mode == "plain":
    files=glob.glob(os.path.join(args.input_dir,"*.npz"))
elif args.collect_mode == "pos5":
    files = glob.glob(os.path.join(f"{args.input_dir}/pos*/*.npz"))

def load_data(file_path_list):
    #各リストの初期化
    left_arm_state_list = []
    right_arm_state_list = []
    teleop_left_arm_state_list = []
    teleop_right_arm_state_list = []
    left_img_list = []
    right_img_list = []
    left_hand_img_list = []
    right_hand_img_list = []
    right_digit_img_list = []

    seq_length = []

    # トピックごとにデータを時系列順に並べる
    for file_path in file_path_list:
        print(file_path)
        npz_data = np.load(file_path)

        # 各画像データを切り抜き
        left_imgs = npz_data["left_imgs"][np.newaxis,:,:,:,:] # (B,H,W,C)
        right_imgs = npz_data["right_imgs"][np.newaxis,:,55:-110,205:-55,:]
        # right_imgs = npz_data["right_imgs"][np.newaxis,:,:,:,:]
        left_hand_imgs = npz_data["left_hand_imgs"][np.newaxis,:,:,:,:]
        right_hand_imgs = npz_data["right_hand_imgs"][np.newaxis,:,:,:,:]
        right_digit_imgs = npz_data["right_digit_imgs"][np.newaxis,:,:,:,:]

        # 各関節、画像をappend
        left_arm_state_list.append(npz_data["upperbody_left_arm_states"][:,:])
        right_arm_state_list.append(npz_data["upperbody_right_arm_states"][:,:])
        teleop_left_arm_state_list.append(npz_data["teleop_left_arm_states"])
        teleop_right_arm_state_list.append(npz_data["teleop_right_arm_states"])
        
        left_img_list.append(np.squeeze(resize_img(left_imgs,(64,64)),0))
        right_img_list.append(np.squeeze(resize_img(right_imgs,(64,64)),0))
        left_hand_img_list.append(np.squeeze(resize_img(left_hand_imgs,(64,64),0)))
        right_hand_img_list.append(np.squeeze(resize_img(right_hand_imgs,(64,64)),0))
        right_digit_img_list.append(np.squeeze(resize_img(right_digit_imgs,(64,64)),0))
        seq_length.append((len(left_arm_state_list[-1])))#-1で配列の一番最後（最新のデータ）を取得し、seq_lengthに追加

    # import ipdb;ipdb.set_trace()
    max_seq = max(seq_length)
    print(f"max_seq:{max_seq}")
    
    #list_to_numpyで異なる長さのリストを同じ長さのnumpy配列に変換。max_seqのほうが大きかったらその部分は最後の値で埋める.eipl.utilを参照
    left_arm_states = list_to_numpy(left_arm_state_list, max_seq)
    right_arm_states = list_to_numpy(right_arm_state_list, max_seq)
    teleop_left_arm_states = list_to_numpy(teleop_left_arm_state_list, max_seq)
    teleop_right_arm_states = list_to_numpy(teleop_right_arm_state_list, max_seq)
    left_imgs = list_to_numpy(left_img_list, max_seq)
    right_imgs = list_to_numpy(right_img_list, max_seq)
    left_hand_imgs = list_to_numpy(left_hand_img_list, max_seq)
    right_hand_imgs = list_to_numpy(right_hand_img_list, max_seq)
    right_digit_imgs = list_to_numpy(right_digit_img_list,max_seq)

    return (
        left_arm_states,
        right_arm_states,
        teleop_left_arm_states,
        teleop_right_arm_states,
        left_imgs,
        right_imgs,
        left_hand_imgs,
        right_hand_imgs,
        right_digit_imgs,
    )

    
if __name__ == "__main__":
    # npzファイルをランダムにtrainとtestに振り分ける
    if args.collect_mode == "plain":
        train_data = random.sample(files,int(len(files)*0.8))
        test_data = [x for x in files if x not in train_data]
    elif args.collect_mode == "pos5":
        train_data = [x for x in files if os.path.join(x).split("/")[3] in ["pos1","pos3","pos5"]]
        test_data = [x for x in files if os.path.join(x).split("/")[3] in ["pos2","pos4"]]
    # 正規化用(arm_statesだけで良い)
    (
        left_arm_states,
        right_arm_states,
        teleop_left_arm_states,
        teleop_right_arm_states,
        left_imgs,
        right_imgs,
        left_hand_imgs,
        right_hand_imgs,
        right_digit_imgs,

    ) = load_data(files)

    # trainデータ作成
    (
        train_left_arm_states,
        train_right_arm_states,
        train_teleop_left_arm_states,
        train_teleop_right_arm_states,
        train_left_imgs,
        train_right_imgs,
        train_left_hand_imgs,
        train_right_hand_imgs,
        train_right_digit_imgs,

    ) = load_data(train_data)

    # testデータ作成
    (
        test_left_arm_states,
        test_right_arm_states,
        test_teleop_left_arm_states,
        test_teleop_right_arm_states,
        test_left_imgs,
        test_right_imgs,
        test_left_hand_imgs,
        test_right_hand_imgs,
        test_right_digit_imgs,
        
    ) = load_data(test_data)

    # 以下、時系列の長さとサンプルの数は1_bag2npyですべて揃えているため、left_arm_statesを使用
    data_len = train_left_arm_states.shape[1] # 時系列の長さ
    train_data_num = train_left_arm_states.shape[0] # サンプルの数
    test_data_num = test_left_arm_states.shape[0] 
 
    base_name=os.path.basename(files[0])
    folder_path=args.input_dir.split(f"npy/")[0]
    folder_path = folder_path.rstrip("/npy/") + f"{args.output_num}"
    
    # ディレクトリ作成
    try:
        os.makedirs(f"{folder_path}/train",exist_ok=True)
        os.makedirs(f"{folder_path}/test",exist_ok=True)
        os.makedirs(f"{folder_path}/data",exist_ok=True)
    except FileExistsError as e:
        print(e)
    
    # トピックごとにデータを保存
    # trainデータ
    np.save(f"{folder_path}/train/left_imgs.npy", train_left_imgs.astype(np.uint8))
    np.save(f"{folder_path}/train/right_imgs.npy", train_right_imgs.astype(np.uint8))

    np.save(f"{folder_path}/train/left_arm_states.npy", train_left_arm_states.astype(np.float32))
    np.save(f"{folder_path}/train/right_arm_states.npy", train_right_arm_states.astype(np.float32))
    
    np.save(f"{folder_path}/train/teleop_left_arm_states.npy", train_teleop_left_arm_states.astype(np.float32))
    np.save(f"{folder_path}/train/teleop_right_arm_states.npy", train_teleop_right_arm_states.astype(np.float32))

    np.save(f"{folder_path}/train/right_hand_imgs.npy", train_right_hand_imgs.astype(np.uint8))
    np.save(f"{folder_path}/train/left_hand_imgs.npy", train_left_hand_imgs.astype(np.uint8))
    np.save(f"{folder_path}/train/right_digit_imgs.npy", train_right_digit_imgs.astype(np.uint8))

    #testデータ
    np.save(f"{folder_path}/test/left_imgs.npy", test_left_imgs.astype(np.uint8))
    np.save(f"{folder_path}/test/right_imgs.npy", test_right_imgs.astype(np.uint8))

    np.save(f"{folder_path}/test/left_arm_states.npy", test_left_arm_states.astype(np.float32))
    np.save(f"{folder_path}/test/right_arm_states.npy", test_right_arm_states.astype(np.float32))
    
    np.save(f"{folder_path}/test/teleop_left_arm_states.npy", test_teleop_left_arm_states.astype(np.float32))
    np.save(f"{folder_path}/test/teleop_right_arm_states.npy", test_teleop_right_arm_states.astype(np.float32))

    np.save(f"{folder_path}/test/right_hand_imgs.npy", test_right_hand_imgs.astype(np.uint8))
    np.save(f"{folder_path}/test/left_hand_imgs.npy", test_left_hand_imgs.astype(np.uint8))
    np.save(f"{folder_path}/test/right_digit_imgs.npy", test_right_digit_imgs.astype(np.uint8))

    # 正規化データ
    # ただし、teleop_right_joint_bounds.shape : (2, 9)
    # すべてのデータのうちのmin,maxの配列を求める
    right_joint_bounds = calc_minmax(right_arm_states)
    left_joint_bounds = calc_minmax(left_arm_states)
    teleop_right_joint_bounds = calc_minmax(teleop_right_arm_states)
    teleop_left_joint_bounds = calc_minmax(teleop_left_arm_states)

    # import ipdb;ipdb.set_trace()
    np.save(f"{folder_path}/data/right_joint_bounds",right_joint_bounds)
    np.save(f"{folder_path}/data/left_joint_bounds",left_joint_bounds)
    np.save(f"{folder_path}/data/teleop_right_joint_bounds",teleop_right_joint_bounds)
    np.save(f"{folder_path}/data/teleop_left_joint_bounds",teleop_left_joint_bounds)

    print(f"データ保存完了:")
    print(f"  trainデータ数: {train_data_num}")
    print(f"  testデータ数: {test_data_num}")
    print(f"  時系列長: {data_len}")
    print(f"  保存先: ./data/")
    print(f"  パラメータ保存先: ./param/")    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from IPython.utils.path import target_outdated
import cv2
import matplotlib

import ipdb.stdout
matplotlib.use('Agg')
import sys
import argparse
import time
import numpy as np
import rospy
import torch

import ipdb
import glob
from natsort import natsorted
from scipy.interpolate import CubicSpline

from eipl.utils import normalization
from eipl.utils import restore_args, tensor2numpy, deprocess_img

# ROS
import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Header
# numpyの2系ではcv_bridgeが使えないことに注意
# from cv_bridge import CvBridge, CvBridgeError

from rt_core import RTCore

# local
sys.path.append("../")
from util import Visualize, Processor, Deprocessor, RTSelector


class RTControl(RTCore):
    def __init__(self, args):
        super(RTControl, self).__init__()
        freq = args.freq
        exptime = args.exptime
        
        self.r = rospy.Rate(freq)
        self.nloop = freq * exptime
        self.open_ratio = args.open_ratio
        
        # deviceの設定(basicはcpu only) 
        try:
            if args.device >= 0:
                device = "cuda:{}".format(args.device)
            else:
                device = "cpu"
        except TypeError as e:
            device = "cpu"
        
        # ../log/dir_name/から，args.jsonを読み込み
        log_dir_path = f"../log/{args.log_dir_name}"
        try:
            params = restore_args(os.path.join(log_dir_path, "args.json"))
        except FileNotFoundError as e:
            print("no such file!")
            exit()
        
        minmax = [params["data"]["vmin"], params["data"]["vmax"]]
        stdev = params["data"]["stdev"] * (params["data"]["vmax"] - params["data"]["vmin"])
        
        img_bounds = [0.0,255.0]
        self.arm_state_bounds = np.load(f"../data/param/arm_state_bound.npy")

        self.eye_img_size = params["data"]["eye_img_size"]
        self.processor = Processor(img_bounds, self.arm_state_bounds, minmax)
        model_name = params["model"]["model_name"]

        selector = RTSelector(params, device)
        self.model = selector.select_model()    # モデルを取得

        """
        numpyのエラーが発生する場合, 実行環境のnumpyのバージョンが1系であることが原因
        numpyの2系が入っているvenvのpy310を起動することで回避が可能
        """
        weight_pathes = natsorted(glob.glob(f"{log_dir_path}/*.pth"))
        
        ckpt = torch.load(weight_pathes[args.ckpt_idx], map_location=torch.device("cpu"), weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        
        self.deprocessor = Deprocessor(img_bounds, self.arm_state_bounds, minmax, select_idxs=[0,])
        self.key_dim = params["model"]["key_dim"]
        
        self.beta = args.beta
        
    def set_init_joint_state(self, init_exptime=3, init_freq=100):
        """
        init_arm_posは関節の初期値
        初期値へ移動する際に，すべての関節を同時に動かすと腕が机に当たるため，
        腕の先の関節をはじめに曲げた後，腕の根本の関節を動かしている
        """
        init_arm_pos = np.array([2059, 2062, 2039, 1283, 3303, 2030, 1048, 2064,  294])
        tar1_arm_pos = np.concatenate([self.right_arm_state[:-5], init_arm_pos[-5:]])        
        tar2_arm_pos = init_arm_pos
        
        input("is it okey to move to init pos?")
        self.move_arm(tar_arm_pos=tar1_arm_pos,
                      arm_msg=self.right_arm_msg,
                      arm_pub=self.right_arm_pub,
                      exp_time=init_exptime,
                      freq=init_freq)
        
        input("finish pos1, move pos2")
        self.move_arm(tar_arm_pos=tar2_arm_pos,
                      arm_msg=self.right_arm_msg,
                      arm_pub=self.right_arm_pub,
                      exp_time=init_exptime,
                      freq=init_freq)
    
    """
    ターゲットの関節角度へ移動
    移動中の関節角度はspline or linearで補完している
    """
    def move_arm(self, 
                tar_arm_pos, arm_msg, arm_pub,
                exp_time=3, freq=100,
                interpolate_type="spline"):
        nloop = exp_time * freq

        tar_arm_pos = np.clip(tar_arm_pos, self.arm_state_bounds[0], self.arm_state_bounds[1])
        curr_arm_pos = np.array(arm_msg.position)
        
        if interpolate_type == "linear":
            arm_traj = np.linspace(curr_arm_pos, tar_arm_pos , nloop)
        elif interpolate_type == "spline":
            time_points = np.array([0, exp_time])  # 開始時間と終了時間
            traj_points = np.vstack([curr_arm_pos, tar_arm_pos])
            splines = [
                CubicSpline(time_points, traj_points[:, i], bc_type=((1, 0), (1, 0)))
                for i in range(traj_points.shape[1])
            ]
            time_stamps = np.linspace(0, exp_time, nloop)
            arm_traj = np.array([spline(time_stamps) for spline in splines]).T
            
        for i in range(nloop):        
            arm_msg.header.stamp = rospy.Time.now()
            arm_msg.position = arm_traj[i]
            arm_pub.publish(arm_msg)
            time.sleep(1./freq)    
    
    # bridge.cv2_to_imgmsg の代わりに実装
    def cv2_to_imgmsg(self, out_img, encoding="bgr8"):
        # 画像の幅、高さ、チャンネル数を取得
        height, width, channels = out_img.shape
        
        # Imageメッセージの作成
        img_msg = Image()
        img_msg.header = Header()
        img_msg.height = height
        img_msg.width = width
        img_msg.encoding = encoding
        img_msg.is_bigendian = False
        img_msg.step = channels * width  # 1行のバイト数
        img_msg.data = out_img.tobytes()  # NumPy配列をバイト列に変換
        
        return img_msg
    
    
    def run(self):
        if self.right_arm_state is not None: #right_arm_state(SubscribeしているROSのメッセージ)がNone出ないなら
            self.set_init_joint_state() #初期位置に移動

            input("Are you ready?")
            state_dict = {"key": None, "vec": None, "union": None}
            # state_dict = None
            y_vec_hat_list, y_img1_hat_list = [], []
            
            for loop_ct in range(self.nloop):
                # 明るさ調節用
                # right_img = cv2.convertScaleAbs(self.right_img, alpha=1, beta=self.beta)
                
                right_img = self.right_img#right_img(H,W,C)
                rt_right_eye_imgs = np.expand_dims(right_img, axis=[0,1])#rt_right_eye_imgs(B,S,C,H,W)で(1,1,3,64,64)
                rt_vecs = np.expand_dims(np.clip(self.right_arm_state, 
                                                 self.arm_state_bounds[0], 
                                                 self.arm_state_bounds[1]), axis=[0,1])
                
                _rt_right_eye_imgs = self.processor.process_imgs(rt_right_eye_imgs, resize=self.eye_img_size)
                _rt_vecs = self.processor.process_vecs(rt_vecs)
                
                x_eye_img1 = torch.Tensor(_rt_right_eye_imgs[:,0]).to(torch.float32)
                x_vec = torch.Tensor(_rt_vecs[:,0]).to(torch.float32)
                
                # 前stepで予測した値を一定割合混ぜ合わせている
                if loop_ct > 0:
                    # x_eye_img1 = self.open_ratio * x_eye_img1 + (1.0 - self.open_ratio) * y_img1_hat_list[-1]
                    x_vec = self.open_ratio * x_vec + (1.0 - self.open_ratio) * y_vec_hat_list[-1]
                
                (y_eye_img1_hat,
                 y_vec_hat, 
                 enc_eye_key1, dec_eye_key1, 
                 enc_eye_key_map1,
                 state_dict) = self.model(x_eye_img1, x_vec, state_dict)
                
                y_vec_hat_list.append(y_vec_hat)
                y_img1_hat_list.append(y_eye_img1_hat)
                
                y_eye_img1_hat = y_eye_img1_hat.unsqueeze(dim=1)
                y_vec_hat = y_vec_hat.unsqueeze(dim=1)
                
                enc_eye_key1 = enc_eye_key1.unsqueeze(dim=1)
                dec_eye_key1 = dec_eye_key1.unsqueeze(dim=1)
                
                enc_eye_key1 = self.deprocessor.deprocess_key(enc_eye_key1, self.eye_img_size)[0,0]
                dec_eye_key1 = self.deprocessor.deprocess_key(dec_eye_key1, self.eye_img_size)[0,0]
                
                pred_eye_img1 = self.deprocessor.deprocess_img(y_eye_img1_hat)[0,0].copy()
                pred_vec = self.deprocessor.deprocess_vec(y_vec_hat)[0,0].copy()
                
                curr_eye_img1 = self.deprocessor.deprocess_img(x_eye_img1.unsqueeze(1))[0,0].copy()
                curr_vec = self.deprocessor.deprocess_vec(x_vec.unsqueeze(1))[0,0].copy()
                
                for i in range(self.key_dim):
                    cv2.circle(curr_eye_img1, tuple(enc_eye_key1[i]), 1, (255,0,0), thickness=-1)
                    cv2.circle(curr_eye_img1, tuple(dec_eye_key1[i]), 1, (0,0,255), thickness=-1)
                
                out_img = np.concatenate((curr_eye_img1, pred_eye_img1), axis=1)
                # out_img_msg = self.bridge.cv2_to_imgmsg(out_img, encoding="bgr8")
                out_img_msg = self.cv2_to_imgmsg(out_img)
                self.pred_img_pub.publish(out_img_msg)

                self.right_arm_msg.header.stamp = rospy.Time.now()
                tar_right_pos = np.clip(np.round(pred_vec), self.arm_state_bounds[0], self.arm_state_bounds[1])
                
                self.right_arm_msg.position = tar_right_pos
                
                # ipdb.set_trace()
                if loop_ct > 10:
                    self.right_arm_pub.publish(self.right_arm_msg)
                
                print(f"{loop_ct}: pred: {np.round(pred_vec)[-1]}")
                self.r.sleep()
        else:
            print("check joint publish!")
            exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir_name", type=str, default="20250407_2036_12")
    parser.add_argument("--freq", type=int, default=10)
    parser.add_argument("--exptime", type=int, default=13)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--open_ratio", type=float, default=1.0)
    parser.add_argument("--beta", type=int, default=20)
    parser.add_argument("--ckpt_idx", type=int, default=-1)
    args = parser.parse_args()
    
    try:
        rospy.init_node("task_node", anonymous=True)
        task = RTControl(args)
        time.sleep(1)
        task.run()
        sys.exit()
    except rospy.ROSInterruptException or KeyboardInterrupt or EOFError as e:
        init_exptime=3
        init_freq=100
        nloop = init_exptime * init_freq
        
        current_position = np.array(task.upper_msg.position)
        target_position = np.array(task.upper_msg.position)
        target_position[11] = 223
        
        trajectory = np.linspace(current_position, target_position , nloop)
        for i in range(nloop):        
            task.upper_msg.header.stamp = rospy.Time.now()
            task.upper_msg.position = trajectory[i]
            time.sleep(1./init_freq)

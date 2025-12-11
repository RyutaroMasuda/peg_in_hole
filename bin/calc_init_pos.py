#!/usr/bin/env python3
import numpy as np
import glob
import os

# すべてのposディレクトリのnpzファイルを取得
npz_files = glob.glob('./data/20250510/npy/pos*/**/*.npz', recursive=True)
print(f"Found {len(npz_files)} npz files")

all_joints = []

for file in npz_files:
    try:
        data = np.load(file)
        print(f"Processing {file}")
        print(f"Keys: {list(data.keys())}")
        
        # 右腕の関節値を取得
        if 'upperbody_right_arm_states' in data:
            joints = data['upperbody_right_arm_states']
            print(f"  Joints shape: {joints.shape}")
            
             # 1次元 or 2次元配列の場合に対応
            if joints.ndim == 1:
                all_joints.append(joints)
                print(f"  Added 1D joint data: {joints}")
            elif joints.ndim == 2:
                # 最初のフレーム（初期状態）のみを使用
                all_joints.append(joints[0])
                print(f"  Added first frame from 2D joint data: {joints[0]}")
        else:
            print(f"  No 'upperbody_right_arm_states' found in {file}")
            
    except Exception as e:
        print(f"Error processing {file}: {e}")

if all_joints:
    all_joints = np.array(all_joints)
    print(f"\nAll joints shape: {all_joints.shape}")
    mean_joint = np.mean(all_joints, axis=0)
    print(f"\n平均初期関節値: {mean_joint}")
    print(f"平均初期関節値 (rounded): {np.round(mean_joint).astype(int)}")
    
    # bin/rt_control.pyのinit_arm_posにコピペ用の形式
    rounded_joints = np.round(mean_joint).astype(int)
    print(f"\ninit_arm_pos = np.array({list(rounded_joints)})")
else:
    print("No joint data found!")
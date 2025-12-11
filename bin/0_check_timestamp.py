#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import cv2
import glob
import rospy
import rosbag
import argparse
import numpy as np
import ipdb
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
parser.add_argument("bag_dir", type=str)
parser.add_argument("--freq", type=float, default=10)
args = parser.parse_args()


files = glob.glob(os.path.join(args.bag_dir, "*.bag"))
files.sort()
for file in files:
    print(file)
    save_dir_name = file.split(".bag")[0].replace("bag", "fig")
    try:
        os.makedirs(save_dir_name)
    except FileExistsError as e:
        print(e)

    # Open the rosbag file
    bag = rosbag.Bag(file)

    # Get the start and end times of the rosbag file
    start_time = bag.get_start_time()
    end_time = bag.get_end_time()
    
    # Get the topics in the rosbag file
    topics = [
        "/maharo/upperbody_left/joint_states",
        "/maharo/upperbody_right/joint_states",
        
        "/teleop/left_joint_states",
        "/teleop/right_joint_states",
        
        "/zed/zed_node/left/image_rect_color",
        "/zed/zed_node/right/image_rect_color",
    ]
    
    # 各トピックのタイムスタンプを保存する辞書
    topic_timestamps = {topic: [] for topic in topics}

    # 全トピックの全メッセージに対してタイムスタンプを取得
    for topic, msg, t in bag.read_messages(topics=topics):
        topic_timestamps[topic].append(t.to_sec())

    json_data = {}
    
    # 各トピックごとに時間間隔を解析
    for topic, timestamps in topic_timestamps.items():
        if len(timestamps) < 2:
            print(f"[{topic}] メッセージ数が少なすぎるためスキップ")
            continue

        intervals = np.diff(timestamps)
        intervals = intervals[1:]
        print(f"\n[{topic}]")
        print(f"- メッセージ数: {len(timestamps)}")
        print(f"- 平均間隔: {np.mean(intervals):.3f} sec")
        print(f"- 最小間隔: {np.min(intervals):.3f} sec")
        print(f"- 最大間隔: {np.max(intervals):.3f} sec")
        
        json_data[topic] = {
            "msg_num": len(timestamps),
            "ave_interval": np.mean(intervals),
            "min_interval": np.min(intervals),
            "max_interval": np.max(intervals)
        }

        topic_name = topic.replace("/", "_")
        # 簡易プロット（必要なら）
        plt.figure()
        plt.plot(intervals)
        plt.title(f"Interval for {topic}")
        plt.xlabel("Frame index")
        plt.ylabel("Interval (sec)")
        plt.grid(True)
        plt.savefig(f"{save_dir_name}/{topic_name}_timestamp_trend.png")

    with open(f"{save_dir_name}/interval.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    # ipdb.set_trace()
    
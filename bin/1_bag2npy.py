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
from sensor_msgs.msg import CompressedImage


parser = argparse.ArgumentParser()
parser.add_argument("bag_dir", type=str)
parser.add_argument("--freq", type=float, default=10)
parser.add_argument("--collect_mode",type=str,default="pos5")
args = parser.parse_args()
if args.collect_mode == "plain":
    files = glob.glob(os.path.join(args.bag_dir, "*.bag"))
    files.sort()
elif args.collect_mode == "pos5":
    files = glob.glob(os.path.join(f"{args.bag_dir}/pos*/*.bag"))
    import ipdb;ipdb.set_trace()
    files.sort()

for file in files:
    print(file)
    base = os.path.splitext(os.path.basename(file))[0]
    bag_root = os.path.join(args.bag_dir)
    npy_root = bag_root.replace("/bag", "/npy")
    if args.collect_mode == "plain":
        save_dir = npy_root
        savename = os.path.join(save_dir, base + ".npz")
    elif args.collect_mode == "pos5":
        rel_path = os.path.relpath(file, bag_root)
        save_dir = os.path.join(npy_root, os.path.dirname(rel_path))
        savename = os.path.join(save_dir, base + ".npz")
    os.makedirs(save_dir,exist_ok=True)
    # Open the rosbag file
    bag = rosbag.Bag(file)
    # Get the start and end times of the rosbag file
    start_time = bag.get_start_time()
    end_time = bag.get_end_time()
    
    # Get the topics in the rosbag file
    # topics = bag.get_type_and_topic_info()[1].keys()
    topics = [
        # 両手、両目、両手カメラ、右手触覚画像
        "/maharo/left_arm/upperbody/joint_states",
        "/maharo/right_arm/upperbody/joint_states",

        "/maharo/left_arm/upperbody/online_joint_states",
        "/maharo/right_arm/upperbody/online_joint_states",
        
        "/zed2i/zed_node/left/image_rect_color/compressed",
        "/zed2i/zed_node/right/image_rect_color/compressed",

        "/left_hand_camera/image_raw/compressed",
        "/right_hand_camera/image_raw/compressed",

        "/digit/right_gripper/image_raw",
    ]

    # Create a rospy.Time object to represent the current time
    current_time = rospy.Time.from_sec(start_time)

    upperbody_left_arm_state_list = []
    upperbody_right_arm_state_list = []

    teleop_left_arm_state_list = []
    teleop_right_arm_state_list = []

    left_img_list = []
    right_img_list = []

    left_hand_img_list = []
    right_hand_img_list = []

    right_digit_list = []
    
    # Loop through the rosbag file at regular intervals (args.freq)
    freq = 1.0 / float(args.freq)
    while current_time.to_sec() < end_time:
        print(current_time.to_sec())

        # Get the messages for each topic at the current time
        for topic in topics:
            # closest_msg = None
            # closest_time = None
            
            for topic_msg, msg, time in bag.read_messages(topic, start_time=current_time):
                if time >= current_time:
                    if topic == "/maharo/left_arm/upperbody/joint_states":
                        """
                        - left_arm/joint1
                        - left_arm/joint2
                        - left_arm/joint3
                        - left_arm/joint4
                        - left_arm/joint5
                        - left_arm/joint6
                        - left_arm/joint7
                        - left_arm/joint8
                        - left_arm/joint9
                        """
                        upperbody_left_arm_state_list.append(np.array(msg.position))

                    if topic == "/maharo/right_arm/upperbody/joint_states":
                        """
                        - right_arm/joint1
                        - right_arm/joint2
                        - right_arm/joint3
                        - right_arm/joint4
                        - right_arm/joint5
                        - right_arm/joint6
                        - right_arm/joint7
                        - right_arm/joint8
                        - right_arm/joint9
                        """
                        upperbody_right_arm_state_list.append(np.array(msg.position))
                    
                    if topic == "/maharo/left_arm/upperbody/online_joint_states":
                        teleop_left_arm_state_list.append(np.array(msg.position))

                    if topic == "/maharo/right_arm/upperbody/online_joint_states":
                        teleop_right_arm_state_list.append(np.array(msg.position))

                    if topic == "/zed2i/zed_node/left/image_rect_color/compressed":
                        # 以下、コメントアウトは/zed2i/zed_node/left/image_rect_color用
                        #bridge = CvBridge()
                        try:
                            np_arr = np.frombuffer(msg.data, np.uint8)
                            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            #img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                            left_img_list.append(img)
                        except Exception as e:
                            rospy.logerr("Error converting img: %s", e)
                            ipdb.set_trace()
                    
                    if topic == "/zed2i/zed_node/right/image_rect_color/compressed":
                        # bridge = CvBridge()
                        try:
                            np_arr = np.frombuffer(msg.data, np.uint8)
                            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            #img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                            right_img_list.append(img)
                        except Exception as e:
                            rospy.logerr("Error converting img: %s", e)
                            ipdb.set_trace()

                    if topic == "/left_hand_camera/image_raw/compressed":
                        # bridge = CvBridge()
                        try:
                            np_arr = np.frombuffer(msg.data, np.uint8)
                            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            #img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                            left_hand_img_list.append(img)
                        except Exception as e:
                            rospy.logeer("Error converting img: %s", e)
                            ipdb.set_trace()

                    if topic == "/right_hand_camera/image_raw/compressed":
                        # bridge = CvBridge()
                        try:
                            np_arr = np.frombuffer(msg.data, np.uint8)
                            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            #img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                            right_hand_img_list.append(img)
                        except Exception as e:
                            rospy.logeer("Error converting img: %s", e)
                            ipdb.set_trace()

                    if topic == "/digit/right_gripper/image_raw":
                        bridge = CvBridge()
                        try:
                            img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                            right_digit_list.append(img)
                        except Exception as e:
                            rospy.logerr("Error converting img: %s", e)
                            ipdb.set_trace()
                    break
        # Wait for the next interval
        current_time += rospy.Duration.from_sec(freq)

    # Close the rosbag file
    bag.close()

    # Convert list to array
    upperbody_left_arm_states = np.array(upperbody_left_arm_state_list, dtype=np.float32)
    upperbody_right_arm_states = np.array(upperbody_right_arm_state_list, dtype=np.float32)

    teleop_left_arm_states = np.array(teleop_left_arm_state_list, dtype=np.float32)
    teleop_right_arm_states = np.array(teleop_right_arm_state_list, dtype=np.float32)
    
    left_imgs = np.array(left_img_list, dtype=np.uint8)
    right_imgs = np.array(right_img_list, dtype=np.uint8)

    left_hand_imgs = np.array(left_hand_img_list, dtype=np.uint8)
    right_hand_imgs = np.array(right_hand_img_list, dtype=np.uint8)

    right_digit_imgs = np.array(right_digit_list, dtype=np.uint8)

    # ipdb.set_trace()
    # Get shorter lenght
    shorter_length = min(
        len(upperbody_left_arm_states),
        len(upperbody_right_arm_states),
        len(teleop_left_arm_states),
        len(teleop_right_arm_states),
        len(left_imgs),
        len(right_imgs),
        len(left_hand_imgs),
        len(right_hand_imgs),
        len(right_digit_imgs),
    )

    # Trim
    upperbody_left_arm_states = upperbody_left_arm_states[:shorter_length]
    upperbody_right_arm_states = upperbody_right_arm_states[:shorter_length]

    teleop_left_arm_states = teleop_left_arm_states[:shorter_length]
    teleop_right_arm_states = teleop_right_arm_states[:shorter_length]
    
    left_imgs = left_imgs[:shorter_length]
    right_imgs = right_imgs[:shorter_length]

    left_hand_imgs = left_hand_imgs[:shorter_length]
    right_hand_imgs = right_hand_imgs[:shorter_length]

    right_digit_imgs = right_digit_imgs[:shorter_length]
    
    # Save
    np.savez(
        savename,
        upperbody_left_arm_states=upperbody_left_arm_states,
        upperbody_right_arm_states=upperbody_right_arm_states,
        teleop_left_arm_states=teleop_left_arm_states,
        teleop_right_arm_states=teleop_right_arm_states,
        left_hand_imgs=left_hand_imgs,
        right_hand_imgs=right_hand_imgs,
        left_imgs=left_imgs,
        right_imgs=right_imgs,
        right_digit_imgs=right_digit_imgs,
    )

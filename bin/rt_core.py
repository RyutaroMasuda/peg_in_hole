#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import matplotlib
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
from eipl.utils import normalization
from eipl.utils import restore_args, tensor2numpy, deprocess_img

# ROS
import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image, CompressedImage
# numpyの2系ではcv_bridgeが使えないことに注意
# from cv_bridge import CvBridge, CvBridgeError

class RTCore:
    def __init__(self):
        # load initial data
        self.left_img = None
        self.right_img = None
        self.left_arm_state = None
        self.right_arm_state = None
        # self.bridge = CvBridge()
        
        self.left_arm_msg = JointState()
        self.right_arm_msg = JointState()
        
        self.left_arm_pub = rospy.Publisher("/teleop/left_joint_states", JointState, queue_size=1)
        self.right_arm_pub = rospy.Publisher("/teleop/right_joint_states", JointState, queue_size=1)

        self.pred_img_pub = rospy.Publisher("/predicted_image", Image, queue_size=1)
        self.show_img_pub = rospy.Publisher("/show_image", Image, queue_size=1)

        rospy.Subscriber("/maharo/upperbody_left/joint_states", JointState, self.left_arm_state_callback)
        rospy.Subscriber("/maharo/upperbody_right/joint_states", JointState, self.right_arm_state_callback)
        rospy.Subscriber("/zed/zed_node/left/image_rect_color", Image, self.left_img_callback)
        rospy.Subscriber("/zed/zed_node/right/image_rect_color", Image, self.right_img_callback)

        # time.sleep(1)

    def left_img_callback(self, msg):
        # bridge = CvBridge()
        # _left_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        _left_img = self.custom_bridge(msg)     # bridge.imgmsg_to_cv2 の代わりに使用
        _left_img = _left_img[:,100:-100]
        _left_img = _left_img[100:-100,100:-50]
        self.left_img = cv2.resize(_left_img, (64,64))
        
    def right_img_callback(self, msg):
        # bridge = CvBridge()
        # _right_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        _right_img = self.custom_bridge(msg)    # bridge.imgmsg_to_cv2 の代わりに使用
        _right_img = _right_img[:,100:-100]
        self._right_img = _right_img[100:-100,100:-50]
        self.right_img = cv2.resize(self._right_img, (64,64))

    def left_arm_state_callback(self, msg):
        if len(self.left_arm_msg.name) == 0:
            self.left_arm_msg.name = msg.name
            self.left_arm_msg.position = msg.position
        self.left_arm_state = np.array(msg.position)
    
    def right_arm_state_callback(self, msg):
        if len(self.right_arm_msg.name) == 0:
            self.right_arm_msg.name = msg.name
            self.right_arm_msg.position = msg.position
        self.right_arm_state = np.array(msg.position)
    
    # bridge.imgmsg_to_cv2 の代わりに実装
    def custom_bridge(self, msg):
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        height, width = msg.height, msg.width
        
        # 画像のデコード（msg.encoding に応じて変換方法を変更する）
        if msg.encoding == "rgb8":
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV は BGR 形式
            return img
        elif msg.encoding == "bgr8":
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
        elif msg.encoding == "mono8":
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            return img
        elif msg.encoding == "bgra8":
            img = np_arr.reshape((height, width, 4))  # BGRA形式
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # αチャンネルを削除してBGRに変換
            return img
        else:
            rospy.logerr(f"Unsupported encoding: {msg.encoding}")
            return

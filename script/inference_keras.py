#!/usr/bin/env python
import rospy
from bayes_people_tracker.msg import PeopleTracker

import os
import pickle
import argparse

import numpy as np
import tensorflow as tf
from ts_model import Model

from geometry_msgs.msg import Pose, PoseArray


class LSTM_Deploy:

    def __init__(self):

        self.INPUT_DIM = 4
        self.RNN_SIZE = 128
        self.PREDICT_STEP = 3

        rospy.init_node('trajectory_listener', anonymous=True)

        self.count = 0
        self.pedestrain = []
        self.pedestrain_state = []
        self.current_pose = []
        self.predict_pose = []

        [self.sess, self.model] = self.load_model()

        self.sub = rospy.Subscriber("/people_tracker/positions_throttle", PeopleTracker, self.callback)
        self.pub = rospy.Publisher("/ILIAD/predicted_pose", PoseArray, queue_size=1)
        rospy.spin()

    def load_model(self):

        # Load the saved arguments to the model from the config file
        with open(os.path.join('demo', 'config.pkl'), 'rb') as f:
            saved_args = pickle.load(f)

        # Initialize with the saved args
        model = Model(saved_args, True)
        # Initialize TensorFlow session
        sess = tf.InteractiveSession()
        # Initialize TensorFlow saver
        saver = tf.train.Saver()

        # Get the checkpoint state to load the model from
        #ckpt = tf.train.get_checkpoint_state('save_lstm')
        #print('loading model: ', ckpt.model_checkpoint_path)

        # Restore the model at the checpoint
        saver.restore(sess, 'demo/model.ckpt')

        return sess, model

    def infer(self, pid, pose):
        index = self.pedestrain.index(pid)
        obs = self.current_pose[index]
        state = self.pedestrain_state[index]

        predict, new_state = self.model.infer(self.sess, obs[np.newaxis, :], state, num=self.PREDICT_STEP)
        self.predict_pose[index] = predict[-1]
        self.pedestrain_state[index] = new_state
        # self.predict_pose[index] = obs

    def insert(self, pid, pose):

        self.pedestrain.append(pid)
        self.pedestrain_state.append((np.zeros((1, self.RNN_SIZE)), np.zeros((1, self.RNN_SIZE))))
        self.current_pose.append(np.array([[float(pose.position.x), float(pose.position.y), float(pose.orientation.z), float(pose.orientation.w)]]))
        self.predict_pose.append(np.zeros((1, self.RNN_SIZE)))

    def update(self, pid, pose):

        index = self.pedestrain.index(pid)
        self.current_pose[index] = np.array([[float(pose.position.x), float(pose.position.y), float(pose.orientation.z), float(pose.orientation.w)]])


    def free(self, ped):

        index = self.pedestrain.index(ped)
        del self.pedestrain[index]
        del self.pedestrain_state[index]
        del self.current_pose[index]
        del self.predict_pose[index]

    def publish(self, pid):

        poses_array = PoseArray()

        poses_array.header.frame_id = "odom"
        poses_array.header.stamp = rospy.get_rostime()

        for pid, predict in zip(self.pedestrain, self.predict_pose):
            pose = Pose()
            pose.position.x = predict[0]
            pose.position.y = predict[1]
            pose.position.z = 1.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0

            predict[2:4] = predict[2:4] / (np.dot(predict[2:4], predict[2:4]) ** 0.5)

            pose.orientation.z = predict[2]
            pose.orientation.w = predict[3]

            poses_array.poses.append(pose)

        print poses_array
        self.pub.publish(poses_array)


    def callback(self, people_msg):

        pids = people_msg.pids
        poses = people_msg.poses

        for pid, pose in zip(pids, poses):

            if pid not in self.pedestrain:
                self.insert(pid, pose)
            else:
                self.update(pid, pose)

            self.infer(pid, pose)
            self.publish(pid)


        for ped in self.pedestrain:
            if ped not in pids:
                self.free(ped)

        print "infer " + str(self.count) + "th trajectory message to the file ..."



        self.count += 1


if __name__ == '__main__':
	lstm_deploy = LSTM_Deploy()

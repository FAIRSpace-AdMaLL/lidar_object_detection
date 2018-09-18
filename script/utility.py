# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2017-05-31 13:13:47
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2017-05-31 14:10:51

import numpy as np
import math
import tf


def get_per_class_accuracy(preds, labels, num_cat):

	preds = np.array(preds)
	labels = np.array(labels)

	acc = []

	for c in range(num_cat):
		if np.sum(labels==c) == 0:
			acc.append(1)
		else:
			acc.append(float(np.sum(preds[np.nonzero(labels==c)] == labels[np.nonzero(labels==c)])) / float(np.sum(labels==c)))

	return acc


def convert2images(Cloud, maxn=150):

	Image = []

	for cloud in Cloud:

		n = len(cloud)

		if n > maxn:
			samples = np.random.choice(n, maxn, replace=False) # down-sampling
		else:
			samples = np.random.choice(n, maxn, replace=True) # up-sampling

		cloud = np.asarray(cloud)

		centroid = np.mean(cloud, axis=0)

		img = cloud[samples, :] - centroid
		Image.append(img)

	Image = np.expand_dims(np.asarray(Image, dtype=np.float16), axis=3)

	return Image



def eular2RotationMat(eular_z, eular_y, eular_x):
	#     The rotation matrix R can be constructed as follows by
	#     ct = [cx cy cz] and st = [sx sy sz]
	#
	#     R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
	#            cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
	#              -sy            cy*sx             cy*cx]
	ct = np.asarray([math.cos(eular_z), math.cos(eular_y), math.cos(eular_x)], dtype=np.float32)
	st = np.asarray([math.sin(eular_z), math.sin(eular_y), math.sin(eular_x)], dtype=np.float32)

	R = np.zeros([3, 3], dtype=np.float32)

	R[0, 0] = ct[1] * ct[0];
	R[0, 1] = st[2] * st[1] * ct[0] - ct[2] * st[0];
	R[0, 2] = ct[2] * st[1] * ct[0] + st[2] * st[0];
	R[1, 0] = ct[1] * st[0];
	R[1, 1] = st[2] * st[1] * st[0] + ct[2] * ct[0];
	R[1, 2] = ct[2] * st[1] * st[0] - st[2] * ct[0];
	R[2, 0] = -st[1];
	R[2, 1] = st[2] * ct[1];
	R[2, 2] = ct[2] * ct[1];

	return R


def transformPose(position, orientation, d, theta):
	# position is (x, y, z) in world coordination system
	# orientation is the quanterion (qx, qy, qz, w)
	# d is the distance between robot and pedestrain
	# theta is the angle of pedestrain in x-y plane, anti-clock direction

	trans_mat = tf.transformations.translation_matrix(position)
	rot_mat = tf.transformations.quaternion_matrix(orientation)
	mat1 = np.dot(trans_mat, rot_mat)

	ct = math.cos(theta)
	st = math.sin(theta)
	tx = ct * d
	ty = st * d

	mat2 = np.array([[ct, -st, 0, tx], [st, ct, 0, ty], [0, 0, 1, 1], [0, 0, 0, 1]])

	mat = np.dot(mat1, mat2)

	position2 = tf.transformations.translation_from_matrix(mat)
	orientation2 = tf.transformations.quaternion_from_matrix(mat)

	d2 = np.sqrt(np.sum((position2[0:2] - position[0:2]) ** 2))

	return position2, orientation2


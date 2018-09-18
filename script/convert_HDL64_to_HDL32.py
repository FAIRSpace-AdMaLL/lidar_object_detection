#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2017-12-29 16:55:21
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2017-12-29 16:55:21

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import os, fnmatch
from loam_velodyne.srv import Convert64To32
from sensor_msgs import point_cloud2
import numpy as np
import pcl


def proj_to_velo(calib_data):
	"""Projection matrix to 3D axis for 3D Label"""
	rect = calib_data["R0_rect"].reshape(3, 3)
	velo_to_cam = calib_data["Tr_velo_to_cam"].reshape(3, 4)
	inv_rect = np.linalg.inv(rect)
	inv_velo_to_cam = np.linalg.pinv(velo_to_cam[:, :3])
	return np.dot(inv_velo_to_cam, inv_rect)


def read_calib_file(calib_path):
	"""Read a calibration file."""
	data = {}
	with open(calib_path, 'r') as f:
		for line in f.readlines():
			if not line or line == "\n":
				continue
			key, value = line.split(':', 1)
			try:
				data[key] = np.array([float(x) for x in value.split()])
			except ValueError:
				pass
	return data


def transform_bbox(place, size, rotate, proj_velo):
	rotate = np.pi / 2 - rotate
	place = np.dot(np.asarray(place), np.asarray(proj_velo).transpose())
	place += 0.27
	place = place.tolist()

	return place, size, rotate


def get_boxcorners(place, rotate, size):
	"""Create 8 corners of bounding box from bottom center."""

	x, y, z = place
	h, w, l = size

	corner = np.array([
		[x - l / 2., y - w / 2., z],
		[x + l / 2., y - w / 2., z],
		[x - l / 2., y + w / 2., z],
		[x - l / 2., y - w / 2., z + h],
		[x - l / 2., y + w / 2., z + h],
		[x + l / 2., y + w / 2., z],
		[x + l / 2., y - w / 2., z + h],
		[x + l / 2., y + w / 2., z + h],
	])

	corner -= np.array([x, y, z])

	rotate_matrix = np.array([
		[np.cos(rotate), -np.sin(rotate), 0],
		[np.sin(rotate), np.cos(rotate), 0],
		[0, 0, 1]
	])

	a = np.dot(corner, rotate_matrix.transpose())
	a += np.array([x, y, z])

	return a


def crop_box(in_cloud, bbox_min, bbox_max):
	out_cloud = []

	for pt in in_cloud:
		if pt[0] < bbox_min[0] or pt[1] < bbox_min[1] or pt[2] < bbox_min[2] or pt[0] > bbox_max[0] or pt[1] > bbox_max[
			1] or pt[2] > bbox_max[2]:
			continue
		else:
			out_cloud.append(pt)

	return out_cloud


def get_label(str):
	if str == 'Car':
		return 1
	elif str == 'Pedestrian': # or str == 'Van' or str == 'Truck'
		return 2
	elif str == 'Cyclist':
		return 3
	else:
		return 0


def save_bbox(pc32_array, file_id):
	global label_dir
	global bbox_dir
	global folder_dir
	global calib_dir

	calib = read_calib_file(os.path.join(calib_dir, file_id + '.txt'))
	proj_velo = proj_to_velo(calib)[:, :3]

	read_stream = open(os.path.join(label_dir, file_id + '.txt'), 'r').read().splitlines()
	write_stream = open(os.path.join(folder_dir, 'index.txt'), 'a')

	for i in range(0, len(read_stream)):
		info = read_stream[i].split(' ')
		label = get_label(info[0])
		size_x = float(info[8])
		size_y = float(info[9])
		size_z = float(info[10])
		pos_x = float(info[11])
		pos_y = float(info[12])
		pos_z = float(info[13])

		size = [size_x, size_y, size_z]
		pos = [pos_x, pos_y, pos_z]
		rot = float(info[14])

		pos, size, rot = transform_bbox(pos, size, rot, proj_velo)

		if (pos[0]**2 + pos[1]**2 + pos[2]**2)**0.5 > 40. and label == 2:
			print "abandon this example. Distance to the sensor: " + str((pos[0]**2 + pos[1]**2 + pos[2]**2)**0.5)
			continue
		else:
			print "save this example. Distance to the sensor: " + str((pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2) ** 0.5)

		corners = get_boxcorners(pos, rot, size)

		bbox_min = np.min(corners, axis=0)
		bbox_max = np.max(corners, axis=0)

		if label > 0:
			obj = crop_box(pc32_array, bbox_min, bbox_max)
			obj_np = np.array(obj, dtype=np.float32)
			obj_np = obj_np - np.mean(obj_np, axis=0)

			if len(obj) > 10:
				pc = pcl.PointCloud(obj_np)
				pc_file = os.path.join(bbox_dir, file_id + '_' + str(i) + '.pcd')
				pcl.save(pc, pc_file)

				write_stream.write("%s %s\n" % ((file_id + '_' + str(i) + '.pcd'), str(label)))


def convert64to32(file_id):
	global dataset_dir
	global converted_dir

	pc64_np = np.fromfile(os.path.join(dataset_dir, file_id + '.bin'), dtype=np.float32).reshape(-1, 4)

	header = Header()
	header.stamp = rospy.Time.now()

	pointFiledx = PointField('x', 0, 7, 1)
	pointFiledy = PointField('y', 4, 7, 1)
	pointFieldz = PointField('z', 8, 7, 1)
	pointFieldi = PointField('intensity', 12, 7, 1)
	pointFiled = [pointFiledx, pointFiledy, pointFieldz, pointFieldi]
	pc64_msg = point_cloud2.create_cloud(header, pointFiled, pc64_np)

	CONVERSION_SRV = '/convert64to32_server'

	pc32_array = []
	try:
		rospy.loginfo("call convert64to32 server ...")
		convert_srv = rospy.ServiceProxy(CONVERSION_SRV, Convert64To32)
		res = convert_srv(pc64_msg)

		pc32_msg = res.cloud_out

		for point in point_cloud2.read_points(pc32_msg, skip_nans=True, field_names=('x', 'y', 'z')):
			pc32_array.append([point[0], point[1], point[2]])

		pc = pcl.PointCloud(np.array(pc32_array, dtype=np.float32))
		pc_file = os.path.join(converted_dir, file_id + '.pcd')
		pcl.save(pc, pc_file)
		print "done, saved!"

	except rospy.ServiceException, e:
		print "Service call failed: %s" % e

	return pc32_array


def main():
	rospy.init_node('conversion_node', anonymous=True)

	global dataset_dir
	global label_dir
	global bbox_dir
	global folder_dir
	global calib_dir
	global converted_dir

	dataset_dir = '/home/kevin/data/kitti/training/velodyne'
	converted_dir = '/home/kevin/data/kitti/training/velodyne32'
	calib_dir = '/home/kevin/data/kitti/training/calib'
	folder_dir = '/home/kevin/data/kitti/training'
	label_dir = '/home/kevin/data/kitti/training/label_2'
	bbox_dir = '/home/kevin/data/kitti/training/bbox'


	if not os.path.exists(converted_dir):
		os.makedirs(converted_dir)

	if not os.path.exists(bbox_dir):
		os.makedirs(bbox_dir)

	file_list = os.listdir(dataset_dir)
	pattern = "*.bin"

	for file in file_list:
		if fnmatch.fnmatch(file, pattern):
			print (file)
			tmp = file.split('.')
			file_id = tmp[0]

			pc32_array = convert64to32(file_id)
			save_bbox(pc32_array, file_id)

	rospy.spin()


if __name__ == '__main__': main()

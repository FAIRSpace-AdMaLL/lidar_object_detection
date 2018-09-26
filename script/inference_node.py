#!/usr/bin/env python

# ROS
import rospy
import message_filters
from lidar_object_detection.msg import ClusterArray
from visualization_msgs.msg import Marker, MarkerArray

from geometry_msgs.msg import PoseStamped, PoseArray, Pose
import tf2_geometry_msgs
import tf2_ros

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField

from spencer_tracking_msgs.msg import DetectedPersons, DetectedPerson

# Utility
import time, os, pickle
import utility
import numpy as np
import random

# TF
import tensorflow as tf
from tf_model import Model

# PCL
# import pcl


class Inference:

	def __init__(self):

		rospy.init_node('inference', anonymous=True)

		VELODYNE_SUB_TOPIC = rospy.get_param('~velodyne_topic', '/velodyne_points')
		CLUSTER_SUB_TOPIC = rospy.get_param('~cluster_sub', '/adaptive_clustering/clusters')
		
		MARKER_SUB_TOPOC = rospy.get_param('~marker_sub', '/adaptive_clustering/markers')
		MARKER_PUB_TOPOC = rospy.get_param('~marker_pub', '/adaptive_clustering/object_markers')
		STATIC_VELODYNE_PUB_TOPIC = rospy.get_param('~static_velodyne_topic', '/static_velodyne_points')
		DETECTED_PERSONS_PUB_TOPIC = rospy.get_param('~detected_persons_pub', 'detected_persons')

		self.tf_model_path = rospy.get_param('~tf_model_path', 'saved_models')
		self.sensor_frame_id = rospy.get_param('~sensor_frame_id', 'velodyne') #velodyne_front
		self.target_frame = rospy.get_param('~target_frame', 'odom') 

		self.is_save_bbox = rospy.get_param('~is_save_bbox', False) 
		
		print "VELODYNE_SUB_TOPIC: " + VELODYNE_SUB_TOPIC
		print "CLUSTER_SUB_TOPIC: " + CLUSTER_SUB_TOPIC
		print "MARKER_SUB_TOPOC: " + MARKER_SUB_TOPOC
		print "MARKER_PUB_TOPOC: " + MARKER_PUB_TOPOC
		print "tf model path: " + self.tf_model_path
		print "sensor frame id: " + self.sensor_frame_id
		print "self.target_frame: " + self.target_frame

		self.load_model()

		self.velodyne_sub = message_filters.Subscriber(VELODYNE_SUB_TOPIC, PointCloud2, queue_size=1)
		self.cluster_sub = message_filters.Subscriber(CLUSTER_SUB_TOPIC, ClusterArray, queue_size=1)
		self.marker_sub = message_filters.Subscriber(MARKER_SUB_TOPOC, MarkerArray, queue_size=1)
		#self.pose_sub = message_filters.Subscriber(POSE_SUB_TOPIC, PoseArray, queue_size=1)

		ts = message_filters.ApproximateTimeSynchronizer([self.velodyne_sub, self.cluster_sub, self.marker_sub], 1, 1, 1)
		ts.registerCallback(self.infer)

		self.marker_pub = rospy.Publisher(MARKER_PUB_TOPOC, MarkerArray, queue_size=1)
		self.velodyne_pub = rospy.Publisher(STATIC_VELODYNE_PUB_TOPIC, PointCloud2, queue_size=1)
		self.detected_persons_pub = rospy.Publisher(DETECTED_PERSONS_PUB_TOPIC, DetectedPersons, queue_size=1)

		self.tf_buffer = tf2_ros.Buffer()
		self.tf2_listener = tf2_ros.TransformListener(self.tf_buffer)

		rospy.spin()

	def convertPLmsg2array(self, velodyne_points, mode='XYZI'):
		pcl_array = []

		if mode == 'XYZ':
			for point in point_cloud2.read_points(velodyne_points, skip_nans=True, field_names=("x", "y", "z")):
				pcl_array.append([point[0], point[1], point[2]])
		elif mode == 'XYZI':
			for point in point_cloud2.read_points(velodyne_points, skip_nans=True, field_names=("x", "y", "z", "intensity")):
				pcl_array.append([point[0], point[1], point[2], point[3]])
		else:
			for point in point_cloud2.read_points(velodyne_points, skip_nans=True, field_names=("x", "y", "z", "intensity", "ring")):
				pcl_array.append([point[0], point[1], point[2], point[3], point[4]])

		return pcl_array


	def crop_box(self, in_cloud, marker):

		bbox_min = marker.points[10]
		bbox_max = marker.points[0]

		offset = 0.5

		out_cloud = []

		for pt in in_cloud:
			if pt[0] < bbox_min.x - offset or pt[1] < bbox_min.y - offset or pt[2] < bbox_min.z - offset or pt[0] > bbox_max.x + offset or pt[1] > bbox_max.y + offset or pt[2] > bbox_max.z + offset:
				out_cloud.append(pt)

		return out_cloud

	def transform_3d_pt(self, pt, transform):

		tmp_pose = PoseStamped()
		tmp_pose.pose.position.x = pt[0]
		tmp_pose.pose.position.y = pt[1]
		tmp_pose.pose.position.z = pt[2]

		pose_transformed = tf2_geometry_msgs.do_transform_pose(tmp_pose, transform)

		return [pose_transformed.pose.position.x, pose_transformed.pose.position.y, pose_transformed.pose.position.z]


	def infer(self, velodyne_points, cluster_array, marker_array):

		global fout
		global is_save
		global frame_count

		start_time = time.time()

		# convert PointCloud2 message to PCL
		static_cloud = self.convertPLmsg2array(velodyne_points, mode='XYZI')

		clusters = cluster_array.clusters

		Cloud = []
		for cluster in clusters:
			cloud = self.convertPLmsg2array(cluster)
			cloud_np = np.asarray(cloud)
			# nomalise the cluser and intensity
			cloud_centriod = np.mean(cloud_np[:, :3], axis=0)
			cloud_np[:, :3] -= cloud_centriod
			cloud_np[:, -1] = cloud_np[:, -1] / 255.
			Cloud.append(cloud_np)
			# print cloud_np

		Image = utility.convert2images(Cloud, self.saved_args.input_dim[0])

		feed = {self.model.input_data: Image}
		output, feature, pred = self.sess.run([self.model.output, self.model.feature, self.model.pred], feed)

		# index = np.nonzero(np.max(np.array(output, dtype=np.float32), axis=1) > 0.7)
		# pred = np.array(pred, dtype=np.float32)
		# pred = pred[index[0]]
		# pred = pred.tolist()

		# transform detection to /map
		done = False

		while not done and self.is_save_bbox:
			try:
				transform = self.tf_buffer.lookup_transform(self.target_frame, self.sensor_frame_id, rospy.Time(), rospy.Duration(10))
				done = True
			except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
				print "transform2 exception"
				continue

		print "transform done!"

		objs_markers = MarkerArray()
		detected_persons = DetectedPersons()
		detected_persons.header = velodyne_points.header

		for p, m, o, f in zip(pred, marker_array.markers, output, feature):

			r = 0
			g = 0
			b = 0

			if p == 1:
				r = 1.0
				bbox_min = m.points[10]
				bbox_max = m.points[0]

				detectedPerson = DetectedPerson()
				detectedPerson.pose.pose.position.x = (bbox_min.x + bbox_max.x) / 2.0
				detectedPerson.pose.pose.position.y = (bbox_min.y + bbox_max.y) / 2.0
				detectedPerson.pose.pose.position.z = (bbox_min.z + bbox_max.z) / 2.0
				detectedPerson.modality = DetectedPerson.MODALITY_GENERIC_LASER_3D
				detectedPerson.confidence = 1.0
				detectedPerson.detection_id = 0
				detected_persons.detections.append(detectedPerson)
			elif p == 2:
				r = 1.0
           			g = 1.0 # truck
			else:
				r = 0.
				continue

			m.color.r = r
			m.color.g = g
			m.color.b = b

			objs_markers.markers.append(m)

			if p >= 1: #
				static_cloud = self.crop_box(static_cloud, m)

			if self.is_save_bbox:
				bbox_min = m.points[10]
				bbox_max = m.points[0]

				bbox_min_pt = self.transform_3d_pt([bbox_min.x, bbox_min.y, bbox_min.z], transform)
				bbox_max_pt = self.transform_3d_pt([bbox_max.x, bbox_max.y, bbox_max.z], transform)

				output_line = [frame_count] + bbox_min_pt + bbox_max_pt + o.tolist() + f.tolist()

				for i in output_line:
					fout.write("%f "% i)

				fout.write("\n")

		frame_count += 1

		self.marker_pub.publish(objs_markers)
		self.detected_persons_pub.publish(detected_persons)

		if self.velodyne_pub.get_num_connections():

			header = velodyne_points.header

			pointFiledx = PointField('x', 0, 7, 1)
			pointFiledy = PointField('y', 4, 7, 1)
			pointFieldz = PointField('z', 8, 7, 1)
			pointFieldi = PointField('intensity', 12, 7, 1)
			pointFiled = [pointFiledx, pointFiledy, pointFieldz, pointFieldi]

			header = velodyne_points.header
			static_velodyne = point_cloud2.create_cloud(header, pointFiled, static_cloud)

			self.velodyne_pub.publish(static_velodyne)

		print("[inference_node]: runing time = " + str(time.time()-start_time))

	def load_model(self):

		# Load the saved arguments to the model from the config file
		with open(os.path.join(self.tf_model_path, 'config.pkl'), 'rb') as f:
			self.saved_args = pickle.load(f)

		# Initialize with the saved args
		model = Model(self.saved_args, is_training=False)
		# Initialize TensorFlow session
		sess = tf.InteractiveSession()
		# Initialize TensorFlow saver
		saver = tf.train.Saver()

		saver.restore(sess, os.path.join(self.tf_model_path, 'model.ckpt'))

		self.sess = sess
		self.model = model

		

def main():

	global fout
	global is_save
	global frame_count

	frame_count = 0

	is_save = False

	if is_save:
		import argparse
		parser = argparse.ArgumentParser()
		parser.add_argument("--file_name", help="the name of output file")
		args = parser.parse_args()
		file_name = args.file_name

		fout = open(file_name, 'w')

	obj = Inference()



if __name__=="__main__":main()


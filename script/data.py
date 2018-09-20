import numpy as np
import os
import random
import math
import utility


class DataLoader:

	def __init__(self, batch_size=128, mode='train'):
		self.Data = []
		self.Label = []
		self.Image = []

		self.batch_size = batch_size
		self.mode = mode
		self.reset_batch_pointer()

	def load_data(self, data_dir, index_file):
		info = open(index_file, 'r').read().splitlines()

		print "reading " + str(len(info)) + " files ..."

		for line in info:
			[file, label] = line.split(" ")

			# pcd = pcl.PointCloud_PointXYZI()
			# pcd.from_file(os.path.join(data_dir, file))

			# pcd = pcl.load(os.path.join(data_dir, file))

			# cloud = np.asarray(pcd, dtype=np.float32)
			cloud = np.loadtxt(os.path.join(data_dir, file))

			self.Data.append(cloud)
			self.Label.append(int(label))

		self.num_batches = len(info) / self.batch_size


	def clean_data(self):
		self.Data = []
		self.Label = []
		self.Image = []


	def convert2images(self, maxn=150):
		for cloud in self.Data:
			n = len(cloud)

			if n > maxn:
				samples = np.random.choice(n, maxn, replace=False)  # down-sampling
				img = cloud[samples, :]
			else:
				samples = np.random.choice(n, maxn, replace=True)  # up-sampling
				img = np.zeros([maxn, 4], dtype=np.float32)
				img[0:n, :] = cloud

			self.Image.append(np.expand_dims(np.asarray(img, dtype=np.float32), axis=2))


	def apply_random_rotate(self, pts_np):
		eular_x = 0
		eular_y = 0
		eular_z = 2 * math.pi * random.random()  # random elur z [0, 2pi]
		rotMat = utility.eular2RotationMat(eular_z, eular_y, eular_x)
		pts_np[:, :3] = pts_np[:, :3].dot(rotMat)

		return pts_np


	def next_batch(self):
		# List of source and target data for the current batch
		x_batch = []
		y_batch = []
		seq_len_batch = []

		# For each sequence in the batch
		for i in range(self.batch_size):
			# Extract the trajectory of the pedestrian pointed out by self.pointer
			imgi = self.Image[self.list[self.pointer]]
			labeli = self.Label[self.list[self.pointer]]

			# if self.mode == 'train':
			# 	imgi[:, :, 0] = self.apply_random_rotate(imgi[:, :, 0])

			x_batch.append(imgi)
			y_batch.append(labeli)

			self.tick_batch_pointer()

		return x_batch, y_batch


	def tick_batch_pointer(self):
		self.pointer += 1

		if self.pointer >= len(self.Image):
			self.pointer -= len(self.Image)


	def reset_batch_pointer(self):
		self.pointer = 0

		self.list = np.arange(len(self.Image))

		if self.mode == 'train':
			np.random.shuffle(self.list)


def main():
	obj = DataLoader()
	obj.load_data('/home/kevin/junk', '/home/kevin/train.txt')
	obj.convert2images()

	x, y = obj.next_batch()
	print x, y


if __name__ == "__main__": main()

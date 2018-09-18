#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2017-12-29 16:55:21
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2017-12-29 16:55:21


import os

def main():


	global folder_dir

	folder_dir = '/home/kevin/data/kitti/training'

	read_stream = open(os.path.join(folder_dir, 'index.txt'), 'r').read().splitlines()

	write_stream1 = open(os.path.join(folder_dir, 'train_index.txt'), 'w')
	write_stream2 = open(os.path.join(folder_dir, 'valid_index.txt'), 'w')

	interval = 5

	for i in range(0, len(read_stream)):
		info = read_stream[i].split(' ')

		if i % interval == 0:
			write_stream2.write("%s %s\n" % (info[0], info[1]))
		else:
			write_stream1.write("%s %s\n" % (info[0], info[1]))


if __name__ == '__main__': main()

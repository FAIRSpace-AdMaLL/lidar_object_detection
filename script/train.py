import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle

from tf_model import Model
from data import DataLoader
import utility


def main():

	parser = argparse.ArgumentParser()
	# =====================================================================================
	# Network Parameters
	parser.add_argument('--mpl_size', type=int, default=[64, 64, 64, 128, 1024],
	                    help='size of multi-layer perception hidden state')
	parser.add_argument('--kernel_size', type=int, default=[[1, 4], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
	                    help='size of convolutional kernel')
	parser.add_argument('--input_dim', type=int, default=[100, 4, 1],
	                    help='dim of input')
	parser.add_argument('--output_dim', type=int, default=3,
	                    help='dim of output')
	# Number of layers parameter
	parser.add_argument('--initializer', type=str, default='uniform',
	                    help='initializer for RNN weights: uniform, xavier, svd')
	# Dimension of the classification layer parameter
	parser.add_argument('--classification_layers', type=int, default=[512, 256],
	                    help='Classification Layer Size')
	parser.add_argument('--keep_prob', type=float, default=0.5,
	                    help='dropout keep probability')
	# Lambda regularization parameter (L2)
	parser.add_argument('--lambda_param', type=float, default=0.005,
	                    help='L2 regularization parameter')
	# =====================================================================================
	#  Training Parameters
	parser.add_argument('--batch_size', type=int, default=128,
	                    help='minibatch size')
	parser.add_argument('--num_epochs', type=int, default=100,
	                    help='number of epochs')
	parser.add_argument('--save_every', type=int, default=10,
	                    help='save frequency')
	# Gradient value at which it should be clipped
	parser.add_argument('--grad_clip', type=float, default=10.,
	                    help='clip gradients at this value')
	# Learning rate parameter
	parser.add_argument('--learning_rate', type=float, default=0.005,
	                    help='learning rate')
	# Decay rate for the learning rate parameter
	parser.add_argument('--decay_rate', type=float, default=.95,
	                    help='decay rate for rmsprop')
	# interval of display
	parser.add_argument('--display', type=float, default=100,
	                    help='set display interval')
	# =====================================================================================
	args = parser.parse_args()
	train(args)


def train(args):

	valid = False

	# Create the data loader object. This object would preprocess the data in terms of
	# batches each of size args.batch_size, of length args.seq_length
	train_data_loader = DataLoader(args.batch_size, mode='train')
	valid_data_loader = DataLoader(args.batch_size, mode='valid')

	# data_loader.load_data('/home/kevin/ncfm_data', '/home/kevin/train.txt')
	train_data_loader.load_data('/home/kevin/data/ncfm/bbox',
	                            '/home/kevin/data/ncfm/train.txt')
	train_data_loader.convert2images(maxn=args.input_dim[0])

	if valid:
		valid_data_loader.load_data('/home/kevin/data/kitti/training/bbox',
	                            '/home/kevin/data/kitti/training/valid_index.txt')
		valid_data_loader.convert2images(maxn=args.input_dim[0])

	# Create a MLP model with the arguments
	model = Model(args)

	# Initialize a TensorFlow session
	with tf.Session() as sess:
		# Add all the variables to the list of variables to be saved
		saver = tf.train.Saver(tf.global_variables())

		# Initialize all the variables in the graph
		sess.run(tf.global_variables_initializer())

		tf.summary.scalar('Realtime loss', model.loss)
		tf.summary.scalar('Realtime learning rate', model.lr)
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter('/home/kevin/.log', sess.graph)

		# For each epoch
		for e in range(args.num_epochs):
			# Assign the learning rate (decayed acc. to the epoch number)
			sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
			# Assign is_training
			# sess.run(tf.assign(model.is_training, True))
			# Reset the pointers in the data loader object
			train_data_loader.reset_batch_pointer()

			# For each batch in this epoch
			train_loss = 0.
			train_counter = 0.
			train_accuracy = 0.
			labels = []
			preds = []


			if valid and e % 1 == 0:

				valid_loss = 0.
				valid_accuracy = 0.
				valid_counter = 0.

				# Assign is_training
				sess.run(tf.assign(model.is_training, False))

				valid_data_loader.reset_batch_pointer()

				for b in range(valid_data_loader.num_batches):
					x, y = valid_data_loader.next_batch()

					start = time.time()

					feed = {model.input_data: x, model.target_data: y}
					loss, pred = sess.run([model.loss, model.pred], feed)

					end = time.time()

					valid_loss += loss
					valid_accuracy += np.sum(np.array(pred) == np.array(y))
					valid_counter += args.batch_size
					labels += y
					preds += pred.tolist()

				print "----------------------------------------------------------------------------------------------------------------------"
				print(
					"validation at epoch {}, valid_loss = {:.6f}, accuracy = {:.3f}, time/batch = {:.6f}"
						.format(
						e,
						valid_loss / valid_counter,
						valid_accuracy / valid_counter,
						end - start))

				print utility.get_per_class_accuracy(preds, labels, args.output_dim)

			# training phase
			#sess.run(tf.assign(model.is_training, True))

			for b in range(train_data_loader.num_batches):
				# Tic
				start = time.time()
				# Get the source and target data of the current batch
				# x has the source data, y has the target data
				x, y = train_data_loader.next_batch()

				feed = {model.input_data: x, model.target_data: y}
				loss, pred, _ = sess.run([model.loss, model.pred, model.train_op], feed)

				# Toc
				end = time.time()
				# Print epoch, batch, loss and time taken
				#train_writer.add_summary(summary, train_step)
				train_loss += loss
				train_accuracy += np.sum(np.array(pred) == np.array(y))
				train_counter += args.batch_size

				if b % args.display == 0:
					print(
						"training {}/{} (epoch {}), train_loss = {:.6f}, accuracy = {:.3f}, time/batch = {:.6f}, learning rate = {:.6f}"
							.format(
							e * train_data_loader.num_batches + b,
							args.num_epochs * train_data_loader.num_batches,
							e,
							train_loss / train_counter,
							train_accuracy / train_counter,
							end - start,
							sess.run(model.lr)))

					train_loss = 0.
					train_counter = 0.
					train_accuracy = 0.

				# Save the model if the current epoch and batch number match the frequency
				if (e) % args.save_every == 0 and (
					(e * train_data_loader.num_batches + b) > 0): #* train_data_loader.num_batches + b
					# Save the arguments int the config file
					with open(os.path.join('saved_models', 'config.pkl'), 'wb') as f:
						pickle.dump(args, f)

					saver.save(sess, 'saved_models/model.ckpt')


if __name__ == '__main__':
	main()

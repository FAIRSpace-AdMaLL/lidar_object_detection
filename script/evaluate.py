import numpy as np
import tensorflow as tf

import os
import pickle
import argparse

from data import DataLoader
from tf_model import Model
import time


def main():
    # Define the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')

    # Read the arguments
    eval_args = parser.parse_args()

    with open(os.path.join('saved_models', 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    # Initialize with the saved args
    model = Model(saved_args, 'eval')
    # Initialize TensorFlow session
    sess = tf.InteractiveSession()
    # Initialize TensorFlow saver
    saver = tf.train.Saver()

    # restore the session
    saver.restore(sess, 'saved_models/model.ckpt')

    data_loader = DataLoader(eval_args.batch_size, mode='eval')

    data_loader.load_data('/home/kevin/ncfm_data', '/home/kevin/test.txt')
    data_loader.convert2images(maxn=saved_args.input_dim[0])


    # Maintain the total_error until now
    accuracy = 0
    total_loss = 0.
    counter = 0.

    for e in range(1):
        # Reset the data pointers of the data loader object
        data_loader.reset_batch_pointer()

        for b in range(data_loader.num_batches):
            start = time.time()
            # Get the source, target data for the next batch
            x, y = data_loader.next_batch()

            feed = {model.input_data: x, model.target_data: y}
            loss, pred = sess.run([model.loss, model.pred], feed)

            accuracy += np.sum(np.array(pred)==np.array(y))
            total_loss += loss
            end = time.time()
            print "Processed batch number : ", b, "out of ", data_loader.num_batches, " batches, time consuming: ", end-start

            counter += data_loader.batch_size

    # Print the mean error across all the batches
    print "Total mean accuracy of the evaluation is ", accuracy / counter
    print "Total mean testing loss of the model is ", total_loss / counter

if __name__ == '__main__':
    main()

"""Wue Lab evaluation script."""

import tensorflow as tf
import numpy as np
import os
from glob import glob
from pandas.core.common import flatten
import matplotlib
import argparse
from tqdm import tqdm
from multiprocessing import Process, Queue
from importlib.machinery import SourceFileLoader
import logging
import pickle

from model.probabilistic_unet import ProbUNet
from utils import training_utils

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def image_list(cf):
    
    img_dir = os.path.join(cf.data_dir, cf.resolution, 'test')
    img, ids = [], []
    patient_dirs = glob(os.path.join(img_dir, 'images', '*'))
    img_ixs = np.random.choice(patient_dirs, cf.sample_size, replace=False)

    for i in range(cf.sample_size):
        # get the first image for the patient
        img_path = img_ixs[i]
        pth_split = img_path.rsplit('/', 1)[-1][:-4]
        ids.append(pth_split)

        image = matplotlib.image.imread(img_path)
        image = image[np.newaxis, np.newaxis, ...]
        image = image[:, :, 26:-26, 26:-26]
        img.append(image)
        
        # Load expert masks
        gt_base_path = img_path.replace('images', 'gt')
        # save all expert masks in specified evaluation directory:
        for l in range(cf.num_graders):
            gt_path = gt_base_path.replace('.png', '_l{}.png'.format(l))
            label = matplotlib.image.imread(gt_path)
            
            if len(label.shape)>2:
                label = rgb2gray(label)
                label = label[26:-26, 26:-26]
                label = label[np.newaxis, np.newaxis, ...].round()
            else:
                label = label[26:-26, 26:-26]
                label = label[np.newaxis, np.newaxis, ...]
            
            img_path = os.path.join(cf.out_dir, 'gt/{}_l{}.npy'.format(pth_split, l))
            np.save(img_path, label)
     
    return img, ids


def write_test_predictions(cf):
    """
    Write samples as numpy arrays.
    :param cf: config module
    :return:
    """
    # do not use all gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = cf.cuda_visible_devices

    data_dir = os.path.join(cf.data_dir, cf.resolution)

    # prepare out_dirs
    if not os.path.isdir(cf.out_dir):
        os.mkdir(cf.out_dir)
        
    image_folder = os.path.join(cf.out_dir, 'images')
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)
    
    gt_folder = os.path.join(cf.out_dir, 'gt')
    if not os.path.isdir(gt_folder):
        os.mkdir(gt_folder)

    plot_folder = os.path.join(cf.out_dir, 'plots')
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    images, ids = image_list(cf)
    
    logging.info('Writing to {}'.format(cf.out_dir))

    # initialize computation graph
    prob_unet = ProbUNet(latent_dim=cf.latent_dim, num_channels=cf.num_channels,
                         num_1x1_convs=cf.num_1x1_convs,
                         num_classes=cf.num_classes, num_convs_per_block=cf.num_convs_per_block,
                         initializers={'w': training_utils.he_normal(),
                                       'b': tf.truncated_normal_initializer(stddev=0.001)},
                         regularizers={'w': tf.contrib.layers.l2_regularizer(1.0),
                                       'b': tf.contrib.layers.l2_regularizer(1.0)})
    x = tf.placeholder(tf.float32, shape=cf.network_input_shape)
    y = tf.placeholder(tf.uint8, shape=cf.label_shape)
    mask = tf.placeholder(tf.uint8, shape=cf.loss_mask_shape)

    with tf.device(cf.gpu_device):
        prob_unet(x, is_training=False, one_hot_labels=cf.one_hot_labels)
        sampled_logits = prob_unet.sample()

    saver = tf.train.Saver(save_relative_paths=True)
    with tf.train.MonitoredTrainingSession() as sess:

        print('EXP DIR', cf.exp_dir)
        latest_ckpt_path = tf.train.latest_checkpoint(cf.exp_dir)
        print('CKPT PATH', latest_ckpt_path)
        saver.restore(sess, latest_ckpt_path)

        for k in tqdm(range(len(images))):
            # save all microscopy scans in specified evaluation directory:
            img = images[k]
            img_id = ids[k]
            
            img_path = os.path.join(cf.out_dir, 'images/{}.npy'.format(img_id))
            np.save(img_path, img)
            
            # sample and save samples in evaluation directory
            for i in range(cf.num_samples):
                sample = sess.run(sampled_logits, feed_dict={x: img})
                sample = np.argmax(sample, axis=1)[:, np.newaxis]
                sample = sample.astype(np.uint8)
                sample_path = os.path.join(cf.out_dir, '{}_sample{}.npy'.format(img_id, i))
                np.save(sample_path, sample)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation step selection.')
    parser.add_argument('--write_samples', dest='write_samples', action='store_true')
    parser.add_argument('--eval_samples', dest='write_samples', action='store_false')
    parser.set_defaults(write_samples=True)
    parser.add_argument('-c', '--config_name', type=str, default='evaluation/wue_lab_eval_config.py',
                        help='name of the python file that is loaded as config module')
    args = parser.parse_args()

    # load evaluation config
    cf = SourceFileLoader('cf', args.config_name).load_module()

    # prepare evaluation directory
    if not os.path.isdir(cf.out_dir):
        os.mkdir(cf.out_dir)

    # log to file and console
    log_path = os.path.join(cf.out_dir, 'eval.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('Logging to {}'.format(log_path))

    if args.write_samples:
        logging.info('Writing samples to {}'.format(cf.out_dir))
        write_test_predictions(cf)
    else:
        logging.info('Evaluating samples from {}'.format(cf.out_dir))
        multiprocess_evaluation(cf)
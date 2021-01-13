"""Wue-Lab evaluation config."""

import os
from model import pretrained_weights

config_path = os.path.realpath(__file__)

#########################################
#                 data      			#
#########################################

data_dir = 'data/wue_lab'
resolution = 'quarter'
label_density = 'gtFine'
num_classes = 2
sample_size = 22
num_graders = 3
one_hot_labels = False
ignore_label = 255


#########################################
#               network 			    #
#########################################

cuda_visible_devices = '0'
cpu_device = '/cpu:0'
gpu_device = '/gpu:0'

patch_size = [128,128]
network_input_shape = (None, 1) +tuple(patch_size)
network_output_shape = (None, num_classes) + tuple(patch_size)
label_shape = (None, 1) + tuple(patch_size)
loss_mask_shape = label_shape

base_channels = 32
num_channels = [base_channels, 2*base_channels, 4*base_channels,
				6*base_channels, 6*base_channels, 6*base_channels, 6*base_channels]

num_convs_per_block = 3

latent_dim = 6
num_1x1_convs = 3
analytic_kl = True
use_posterior_mean = False

#########################################
#             evaluation 			    #
#########################################

num_samples = 6
exp_dir = 'model/pretrained_weights'
out_dir = 'wue_lab_eval_output_dir'
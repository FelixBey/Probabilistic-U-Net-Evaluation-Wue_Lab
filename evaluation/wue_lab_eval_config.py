# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wue-Lab evaluation config."""

import os
from model import pretrained_weights

config_path = os.path.realpath(__file__)

#########################################
#                 data      			#
#########################################

data_dir = '/media/data/home/s370876/data/wue_lab'
resolution = 'quarter'
label_density = 'gtFine'
num_classes = 2
sample_size = 20
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
exp_dir = '/media/data/home/s370876/model/wue_lab_weights_18k'
out_dir = '/media/data/home/s370876/wue_lab_eval_output_dir_18k'
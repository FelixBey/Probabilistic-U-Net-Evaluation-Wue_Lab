"""Segmentation Metrics by Matthias Griebel: https://github.com/matjesg/deepflash2/blob/master/deepflash2/utils.py"""


import numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.spatial.distance import jaccard
from scipy.stats import entropy
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.segmentation import relabel_sequential, watershed
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


"""## Pixelwise Analysis"""


#export
def iou(a,b,threshold=0.5):
    '''Computes the Intersection-Over-Union metric.'''
    a = np.array(a) > threshold
    b = np.array(b) > threshold
    overlap = a*b # Logical AND
    union = a+b # Logical OR
    return np.count_nonzero(overlap)/np.count_nonzero(union)


"""## ROI-wise Analysis"""

def label_mask(mask, threshold=0.5, min_pixel=15, do_watershed=False, exclude_border=False):
    '''Analyze regions and return labels'''
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=2)

    # apply threshold to mask
    # bw = closing(mask > threshold, square(2))
    bw = (mask > threshold).astype(int)

    # label image regions
    label_image = label(bw, connectivity=2) # Falk p.13, 8-“connectivity”.

    # Watershed: Separates objects in image by generate the markers
    # as local maxima of the distance to the background
    if do_watershed:
        distance = ndimage.distance_transform_edt(bw)
        # Minimum number of pixels separating peaks in a region of `2 * min_distance + 1`
        # (i.e. peaks are separated by at least `min_distance`)
        min_distance = int(np.ceil(np.sqrt(min_pixel / np.pi)))
        local_maxi = peak_local_max(distance, indices=False, exclude_border=False,
                                    min_distance=min_distance, labels=label_image)
        markers = label(local_maxi)
        label_image = watershed(-distance, markers, mask=bw)

    # remove artifacts connected to image border
    if exclude_border:
        label_image = clear_border(label_image)

    # remove areas < min pixel
    unique, counts = np.unique(label_image, return_counts=True)
    label_image[np.isin(label_image, unique[counts<min_pixel])] = 0

    # re-label image
    label_image, _ , _ = relabel_sequential(label_image, offset=1)

    return (label_image)


def get_candidates(labels_a, labels_b):
    '''Get candiate masks for ROI-wise analysis'''

    label_stack = np.dstack((labels_a, labels_b))
    cadidates = np.unique(label_stack.reshape(-1, label_stack.shape[2]), axis=0)
    # Remove Zero Entries
    cadidates = cadidates[np.prod(cadidates, axis=1) > 0]
    return(cadidates)

#export
def iou_mapping(labels_a, labels_b):
    '''Compare masks using ROI-wise analysis'''

    candidates = get_candidates(labels_a, labels_b)

    if candidates.size > 0:
        # create a similarity matrix
        dim_a = np.max(candidates[:,0])+1
        dim_b = np.max(candidates[:,1])+1
        similarity_matrix = np.zeros((dim_a, dim_b))

        for x,y in candidates:
            roi_a = (labels_a == x).astype(np.uint8).flatten()
            roi_b = (labels_b == y).astype(np.uint8).flatten()
            similarity_matrix[x,y] = 1-jaccard(roi_a, roi_b)

        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

        return(similarity_matrix[row_ind,col_ind],
               row_ind, col_ind,
               np.max(labels_a),
               np.max(labels_b)
               )
    else:
        return([],
               np.nan, np.nan,
               np.max(labels_a),
               np.max(labels_b)
               )


def calculate_roi_measures(*masks, iou_threshold=.5, **kwargs):
    "Calculates precision, recall, and f1_score on ROI-level"
    labels = [label_mask(m, **kwargs) for m in masks]
    matches_iou, _,_, count_a, count_b = iou_mapping(*labels)
    matches = np.sum(np.array(matches_iou) > iou_threshold)
    precision =  matches/count_a
    recall = matches/count_b
    f1_score = 2 * (precision * recall) / (precision + recall)
    return recall, precision, f1_score
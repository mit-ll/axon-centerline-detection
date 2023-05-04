import numpy as np
from scipy.ndimage import distance_transform_edt
from utils.transforms import Skeletonize

''' Metrics for evaluation. '''

def harmonic_mean(x, y, alpha=0.5, ev=0):
    ''' Harmonic mean with optional non-uniform weighting or correction for chance. '''
    return 1/(alpha/x + (1-alpha)/y)

def dice_score(seg_p, seg_gt, precision=False, recall=False):
    '''
    Inputs:
      - `seg_p`(ndarray): Binary predictions
      - `seg_gt`(ndarray): Binary ground truth segmentation mask
    Returns:
      - The Dice coefficient between predicted and ground truth masks
      - Precision (if specified)
      - Recall (if specified)
    '''
    tp = np.sum(seg_p*seg_gt)
    prec = tp/np.sum(seg_p)
    sens = tp/np.sum(seg_gt)
    dice = harmonic_mean(prec, sens)
    if precision and recall:
        return dice, prec, sens
    elif precision:
        return dice, prec
    elif recall:
        return dice, sens
    else:
        return dice

def cldice_score(seg_p, seg_gt, cl_p=None, cl_gt=None, precision=False, recall=False):
    '''
    Inputs:
      - `seg_p`(ndarray): Binary predicted segmentation
      - `seg_gt`(ndarray): Binary ground truth segmentation
      - `cl_p`(ndarray): Centerline predictions
      - `cl_gt`(ndarray): Centerline ground truth
    Returns:
      - cldice score between the predicted and ground truth masks.
      - Precision (if specified)
      - Recall (if specified)

    If not provided, the centerline prediction and ground truth will be
    computed via skeletonization of the respective segmentation masks.
    '''
    # Obtain centerlines
    if cl_p is None:
        cl_p = Skeletonize(threshold=0.5).apply(seg_p)
    if cl_gt is None:
        cl_gt = Skeletonize(threshold=0.5).apply(seg_gt)
    # Topological precision
    tprec = np.sum(cl_p*seg_gt)/np.sum(cl_p)
    # Topological recall
    tsens = np.sum(seg_p*cl_gt)/np.sum(cl_gt)
    cldice = harmonic_mean(tprec, tsens)
    if precision and recall:
        return cldice, tprec, tsens
    elif precision:
        return cldice, tprec
    elif recall:
        return cldice, tsens
    else:
        return cldice

def rhodice_score(cl_p, cl_gt, rho=3, precision=False, recall=False):
    '''
    Inputs:
      - `cl_p`(ndarray): Centerline predictions
      - `cl_gt`(ndarray): Centerline ground truth
      - `rho`(int): True positive buffer distance
    Returns:
      - Dice coefficient between the predicted and ground truth centerlines
        with any predicted voxels within `rho` of the ground truth counting as
        a true positive.
    '''
    # Compute buffers
    cl_p_rho = np.uint8(distance_transform_edt(1-cl_p) <= rho)
    cl_gt_rho = np.uint8(distance_transform_edt(1-cl_gt) <= rho)
    # rho-precision
    rprec = np.sum(cl_p*cl_gt_rho)/np.sum(cl_p)
    # rho-recall
    rsens = np.sum(cl_p_rho*cl_gt)/np.sum(cl_gt)
    rhodice = harmonic_mean(rprec, rsens)
    if precision and recall:
        return rhodice, tprec, tsens
    elif precision:
        return rhodice, tprec
    elif recall:
        return rhodice, tsens
    else:
        return rhodice

def get_overlap_matrix(labels_p, labels_gt):
    '''
    Computes matrix of proportion of voxels falling into each possible
    combination of predicted and ground truth cluster labels 
    '''
    overlaps = np.zeros((labels_p.max() + 1, labels_gt.max() + 1))
    aligned = np.stack((labels_p.flatten(), labels_gt.flatten()))
    indices, counts = np.unique(aligned, return_counts=True, axis=1)
    N = counts.sum()
    overlaps[tuple(indices)] = counts / N
    return overlaps

def rand_f_score(labels_p, labels_gt, alpha=0.5, merge_score=False, split_score=False):
    '''
    Calculates the Rand F-Score (from ISBI2012), adjusted for chance, of predicted
    and ground truth clusterings. In practice, partitions can be computed from the
    connected components of foreground structures in a segmentation mask.

    See: Arganda-Carreras et al. (2015), Hubert & Arabie (1985)

    Inputs:
      - `labels_p`(ndarray): Predicted cluster labels
      - `labels_gt`(ndarray): Ground truth cluster labels
      - `alpha`(float): Weighting of merge score; split score is implicitly
                        weighted by (1-alpha)
    Returns:
      - Adjusted Rand score of ground truth and predicted partitions
      - Merge score (if specified)
      - Split score (if specified)
    '''
    om = get_overlap_matrix(labels_p, labels_gt)

    col_counts = om.sum(0) #t_j
    row_counts = om.sum(1) #s_i

    t_term = np.square(col_counts).sum()
    s_term = np.square(row_counts).sum()
    p_term = np.square(om).sum()

    # Expected value is product of marginals
    ev = s_term*t_term

    # Probability of predicting same segment for two randomly chosen voxels,
    # given they are in the same ground truth segment (adjusted for chance)
    split = (p_term - ev) / (t_term - ev)
    # Probability that two randomly chosen voxels are in the same ground truth
    # segment, given that they are in the same predicted segment (adjusted for chance)
    merge = (p_term - ev) / (s_term - ev)

    rand = harmonic_mean(split, merge, alpha=alpha)

    if split_score and merge_score:
        return rand, merge, split
    elif split_score:
        return rand, split
    elif merge_score:
        return rand, merge
    else:
        return rand

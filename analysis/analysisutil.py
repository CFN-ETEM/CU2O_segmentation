import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from skimage import measure
###Compute true positives, false positives and false negatives by matching predicted centroids to ground truth centroids within a radius r.
def match_centroids(gt, pred, r=2.5):
    """Return counts TP, FP, FN and list of (d_row, d_col) errors."""
    if len(gt)==0 and len(pred)==0:
        return 0,0,0,[]
    tree = cKDTree(gt)
    dists, idx = tree.query(pred, distance_upper_bound=r)
    pairs = [(i, idx[i], dists[i]) for i in range(len(pred)) if idx[i] < len(gt)]
    if not pairs:
        return 0, len(pred), len(gt), []
    P = len(pred); G = len(gt)
    BIG=1e6
    cost = np.full((P, G), BIG)
    for i, j, d in pairs:
        cost[i, j] = d
    rows, cols = linear_sum_assignment(cost)
    matched = [(r, c, cost[r, c]) for r, c in zip(rows, cols) if cost[r, c] < BIG]
    TP = len(matched)
    FP = len(pred) - TP
    FN = len(gt)   - TP
    return TP, FP, FN
###Get centroids of connected components in a binary mask, filtering out small components.
def centroids(mask):
    lbl = measure.label(mask > 0, connectivity=2)
    props = measure.regionprops(lbl)
    return np.array([p.centroid for p in props if p.area > 2]) 
import numpy as np

def map_d2window(edge_xy, contour):
    """
    This function map calculated d to window we defined in split_window2D.py

    The reasons for this function is that the d is calculated based on the contour_evolve,
        which has integer (x, y) coordinates;

    However, the windows are defined in the floating point (x, y) coordinates.
    Therefore, this function use kdTree to find the closest point in the window 
        and assign the d to the windows for further correlation and causality calculation.
    
    """
    xy_windows = init_xy_window(edge_xy)

    edge_xy = np.vstack(edge_xy)

    indices = find_closest_point(edge_xy, contour)

    d_window_indices = map_ind2wd(edge_xy, indices, xy_windows)

    return d_window_indices

def map_ind2wd(edge_xy, indices, xy_windows):
    """
    Map the indices (of closest point) to the window index
    """
    d_window_indices = []
    for idx in indices:
        window_idx = xy_windows[tuple(edge_xy[idx])]
        d_window_indices.append(window_idx)

    d_window_indices = np.array(d_window_indices)
    return d_window_indices

def find_closest_point(edge_xy, contour):
    """
    Find the closest point in the edge_xy to the contour

    """
    from scipy.spatial import cKDTree

    tree = cKDTree(edge_xy)

    _, indices = tree.query(contour)

    return indices

def init_xy_window(edge_xy):
        """
        Initialize the xy_window dictionary
        """
        xy_windows = {}
        for i, window in enumerate(edge_xy[0]):
            for xy in window:
                xy_windows[tuple(xy)] = i
        return xy_windows


def collect_window_acd(d_indices, ac_d):
    """
    The function to calculate the accumulated d in the window
    
    """
    window_acd = [[] for i in range(np.max(d_indices)+1)]
    for idx, d in zip(d_indices, ac_d):
        window_acd[idx].append(d)
    return window_acd


def calculate_window_d(edge_xy, contour_evolve_stack, ac_d):
    """
    Master function for calculate the window in d 

    """
    xy_windows = init_xy_window(edge_xy)

    edge_xy = np.vstack(edge_xy[0])

    indices = find_closest_point(edge_xy, contour_evolve_stack[...,0])

    d_window_indices = map_ind2wd(edge_xy, indices, xy_windows)

    window_acd = collect_window_acd(d_window_indices, ac_d)
    return window_acd  


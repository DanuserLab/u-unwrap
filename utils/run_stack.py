THETA_NUM = 60     ### a parameter for split the disk angle
R_NUM = 30          ### a parameter for split the disk radius
RASTER_SIZE = 256   ### a parameter for visualization; doesn't have specific meaning
BORDER_PAD = 26     ### a parameter for visualization; doesn't have specific meaning

import numpy as np

import unwrap3D.Mesh.meshtools as unwrap3D_meshtools
import unwrap3D.Segmentation.segmentation as unwrap3D_segmentation
import unwrap3D.Image_Functions.image as unwrap3D_image_fn


import calculate_d as cal_d_fns
import classify_window_d 
from split_window2D import UnwrapWindow
import sys
sys.path.append('./unwrap2D')
import unwrap2D as unwrap2D_fns



"""
Functions to run the stacks
"""
def unwrap2D_all_TF(raw, masks, contour_method = 'parametric_line_flow_2D'):
    """
    This function is to unwrap the 2D image for all the frames in the raw image stack.

    The function will return the unwrapped image stack, the unwrapped mask stack, and the unwrapped window stack.

    """
    
    assert raw.shape == masks.shape, f"AssertionError: raw.shape ({raw.shape}) != masks.shape ({masks.shape})"
    import skimage.morphology as skmorph
    import scipy.ndimage as ndimage
    unwrap_img_stack = []
    unwrap_params_stack = []
    unwrap_mask_stack = []
    contour_evolve_stack = []
    v_out_stack = []
    f_steps_out_stack = []
    bdy_index_stack = []
    for frame in range(raw.shape[0]):

        mask_frame = masks[frame] > 0
        mask_frame = skmorph.binary_closing(mask_frame, skmorph.disk(1))
        mask_frame = ndimage.binary_fill_holes(mask_frame)

        v_out, f_steps_out, v_img_out, bdy_index = unwrap2D_fns.unwrap_2D(raw[frame], 
                                                            mask_frame, 
                                                            relax_tol=1.0e-5,
                                                            relax_niters=25,
                                                            relax_omega=1.,
                                                            debugviz_tri_areadistort = False,
                                                            areadistort_max_iter = 100,
                                                            areadistort_delta_h_bound = 0.5,
                                                            areadistort_stepsize=0.1,
                                                            area_distort_flip_tri=True)
        

        unwrap_params, unwrap_img, unwrap_mask = unwrap2D_fns.resample_disk_grid_to_img_grid(v_out[:,1:], 
                                                                    v_img_out[:,1:], # use only the last two coordinates.
                                                                    f_steps_out, 
                                                                    raw[frame],
                                                                    raster_size=RASTER_SIZE, border_pad=BORDER_PAD) # raster_size is the diameter of the disk, border_pad=black area.
    

    

        contour = v_img_out[bdy_index][:,1:]
        contour_evolve = create_contour_evolve(contour, mask_frame, contour_method)
        
        unwrap_img_stack.append(unwrap_img)
        unwrap_params_stack.append(unwrap_params)
        unwrap_mask_stack.append(unwrap_mask)
        contour_evolve_stack.append(contour_evolve)
        v_out_stack.append(v_out)
        f_steps_out_stack.append(f_steps_out)
        bdy_index_stack.append(bdy_index)
    return unwrap_img_stack, unwrap_params_stack, unwrap_mask_stack, contour_evolve_stack, v_out_stack, f_steps_out_stack, bdy_index_stack


def create_contour_evolve(contour, mask, method = 'parametric_line_flow_2D'):
    
    if method == 'parametric_line_flow_2D':
        assert mask is not None, "AssertionError: mask is None"
        mask_dist_tform = unwrap3D_segmentation.sdf_distance_transform(mask)
        mask_dist_tform_gradients = np.array(np.gradient(mask_dist_tform)).transpose(1,2,0) # 2D gradients. (non-normalized, therefore taking into account curvature)

        contour_evolve = unwrap2D_fns.parametric_line_flow_2D(contour,
                                                        external_img_gradient=mask_dist_tform_gradients, 
                                                        E=None, 
                                                        close_contour=True, 
                                                        fixed_boundary = False, 
                                                        lambda_flow=100, # adjusts the balance.  (decrease this to be more similar to original cell shape ), increase to be more like version 1
                                                        step_size=1, # adjusts the spacing between curves. 
                                                        niters=50, 
                                                        conformalize=False,
                                                        eps=1e-12)
    
    if method == 'conformalized_mean_curvature_flow':
        contour_evolve = unwrap3D_meshtools.conformalized_mean_line_flow(contour,
                                                                E=None, 
                                                                close_contour=True, 
                                                                fixed_boundary = False, 
                                                                lambda_flow=0.05e3, 
                                                                niters=10, 
                                                                topography_edge_fix=False, 
                                                                conformalize=True) 

    return contour_evolve

"""
Functions to split the window stacks

"""
def window_all_TF(unwrap_img_stack, unwrap_params_stack):
    window_stack = []
    for unwrap_img, unwrap_params in zip(unwrap_img_stack, unwrap_params_stack):
        windows = UnwrapWindow(unwrap_img, 
                        unwrap_params, 
                        raster_size=RASTER_SIZE, 
                        border_pad=BORDER_PAD)
        
        window_stack.append(windows)
    return window_stack

"""
Functions to Calculate the intensity in each window 

"""
def calculate_window_intensity(window_stack, 
                                r_num = R_NUM,
                                theta_num = THETA_NUM,
                                layer_n = 5):
    window_intensity_stack = []
    edge_xy_stack= []
    for window in window_stack:
        segment_uv, segment_xy, window_values = window.split_window(r_num = R_NUM, 
                                                                    theta_num = THETA_NUM)
        #_, edge_xy = window.get_layer_window(layer = -1)
        
        _, edge_xy = window.get_layer_window(layer = layer_n)
        window_intensity_stack.append(window_values)
        edge_xy_stack.append(edge_xy)
        
        
    return window_intensity_stack, edge_xy_stack

"""
Functions to calculate d in each window

"""

def init_xy_window(edge_xy):
        """
        Initialize the xy_window dictionary
        """
        xy_windows = {}
        for i, window in enumerate(edge_xy[0]):
            for xy in window:
                xy_windows[tuple(xy)] = i
        return xy_windows

def find_closest_point(edge_xy, contour):
    """
    Find the closest point in the edge_xy to the contour

    """
    from scipy.spatial import cKDTree

    tree = cKDTree(edge_xy)

    _, indices = tree.query(contour)

    return indices



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

def collect_window_acd(d_indices, ac_d, dd):
    """
    The most important function to calculate the accumulated d in the window
    
    """
    window_acd = [[] for i in range(np.max(d_indices)+1)]
    window_d = [[] for i in range(np.max(d_indices)+1)]
    for idx, d in zip(d_indices, ac_d):
        window_acd[idx].append(d)
    
    for idx, d in zip(d_indices, dd):
        window_d[idx].append(d)

    return window_acd, window_d

def calculate_window_d(edge_xy, contour_evolve_stack, ac_d, d):
    """
    Master function for calculate the window of d 

    """
    xy_windows = init_xy_window(edge_xy)

    edge_xy = np.vstack(edge_xy[0])

    indices = find_closest_point(edge_xy, contour_evolve_stack[...,0])

    d_window_indices = map_ind2wd(edge_xy, indices, xy_windows)

    window_acd, window_d = collect_window_acd(d_window_indices, ac_d, d)
    return window_acd, d_window_indices, window_d  

def calculate_all_windowd(contour_evolve_stack, all_edge_xy, reference_n):
    ori_d_stack = []
    window_d_stack = []
    window_acd_stack = []
    d_window_indices_stack = []
    for contour_evolve, edge_xy in zip(contour_evolve_stack, all_edge_xy):
            d, ac_d = cal_d_fns.calculate_d(contour_evolve, reference_n)
            window_acd, d_window_indices, window_d = calculate_window_d(edge_xy, contour_evolve, ac_d, d)

            ori_d_stack.append(d)
            window_d_stack.append(window_d) 
            window_acd_stack.append(window_acd)
            d_window_indices_stack.append(d_window_indices)
    return window_acd_stack, d_window_indices_stack, window_d_stack, ori_d_stack


"""
Plot functions

"""

def plot_window_d(window_acd_stack):
    import matplotlib.pyplot as plt

    d = np.zeros((len(window_acd_stack[0]), len(window_acd_stack))) ## N of windows, N of frames
    for i, frame in enumerate(window_acd_stack):
        for j, window in enumerate(frame):
            d[j,i] = np.mean(window)

    plt.imshow(d, cmap='rainbow')
    plt.xlabel('Frame')
    plt.ylabel('Window')
    plt.colorbar()
    plt.title('d of each window in each frame')
    plt.show()
    return d

def plot_intensity(all_intensity, layer = 1, theta_num = 60):
    import matplotlib.pyplot as plt
    intensity = np.zeros((len(all_intensity[0][:theta_num]), len(all_intensity))) ## N of windows, N of frames
    for i, frame in enumerate(all_intensity):
        s = theta_num * (layer - 1)
        e = theta_num * layer
        for j, window in enumerate(frame[s:e]):
            intensity[j,i] = np.mean(window)

    plt.imshow(intensity, cmap='rainbow')
    plt.xlabel('Frame')
    plt.ylabel('Window')
    plt.colorbar()
    plt.title(f'Intensity of layer{layer} each window in each frame')
    plt.show()
    return intensity





if __name__=="__main__":
    import pylab as plt 
    from matplotlib import cm 
    import os 
    import skimage.io as skio 
    import skimage.segmentation as sksegmentation
    import skimage.morphology as skmorph
    import skimage.measure as skmeasure
    import scipy.ndimage as ndimage
    basefolder = '/Users/yushiqiu/Desktop/UTSW/rotation_project/Gaudenz/track_window'
    datafolder = os.path.join(basefolder,'data/Registration')
    savefolder = os.path.join(basefolder,'output/2024_08_27')

    mask = os.path.join(datafolder, 'MDACell5_Masks.tif')
    raw = os.path.join(datafolder, 'MDACell5_NoRegistration.tif')

    masks = skio.imread(mask)
    raw = skio.imread(raw)

    print(masks.shape)
    print(raw.shape)


    #frame = min(raw.shape[0], masks.shape[0]) ### the number of frames to process because these two has different shape
    frame = 50
    unwrap_img_stack, unwrap_params_stack, unwrap_mask_stack, contour_evolve_stack,v_out_stack, f_steps_out_stack, bdy_index_stack = unwrap2D_all_TF(raw[:frame], masks[:frame], "parametric_line_flow_2D")
    window_stack = window_all_TF(unwrap_img_stack, unwrap_params_stack)
    all_intensity, all_edge_xy = calculate_window_intensity(window_stack)
    window_acd_stack, d_w_id_stack, window_d_stack, ori_d_stack = calculate_all_windowd(contour_evolve_stack, all_edge_xy, reference_n = 5)

    d = plot_window_d(window_acd_stack).T
    intensity = plot_intensity(all_intensity).T
    for i in range(2, 6):
        print(intensity.shape)
        a = plot_intensity(all_intensity, layer = i).T
        intensity = np.concatenate((intensity,a),axis=0)

    layer_1 = intensity[:frame]
    cc_layer1 = calculate_cross_correlation(layer_1, d, lag = [-43,72], center = 43)


    cc_list = []
    for i, c  in enumerate(cc_layer1):
        cc_list.append(c)
    plt.plot([i for i in range(-43,72)], cc_list)
"""
2025_04_08 reorganize code for frame-by-frame registration and boundary points propagation
framework:
Step 1:
    Demon Registration:
        - save 3 channel images to check the registration
        - extract displacement field for boundary propagation
        - validation of the registration

Step 2:
    Boundary Point Propagation:
        - Use displacement field got from Step 1 to update the boundary (x, y) coordinate
        - Map propagated_boundary to the fixed_boundary (directly output of igl.boundary_loop(f))
            --> currently forward euler + GVF 
            -- If there's redundance in one frame --> perform linear interpolation between points
        - Reorder propagated_boundary (using fixed_boundary) to make sure there's no intersection
        - Detect anomalies (displacement too big; order of points wrong)
        

Step 3:
    Interpolation on the UV coordinate
        - Map the first frame boundary points to the UV coordinate
        - Use propagated_boundary to interpolate the UV coordinate of other frames
    

This file contains the Step 2

"""


# from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from copy import deepcopy
import cv2
from tqdm import tqdm
import os

"""
##########################################################################################################################

Below are the functions for generating standard boundaries

"""


def extract_igl_boudnary(mask):
    """
    Create a mesh first and extract boundary from the mesh

    Don't use the skimage.measure.find_contours because the boundaries 
    cannot be mapped to the (UV) coordinate later

    The mask doesn't have to be mask, it can be image data

    """
    import skimage.morphology as skmorph
    import scipy.ndimage as ndimage
    import unwrap3D.Mesh.meshtools as unwrap3D_meshtools
    import igl
    
    mask = skmorph.binary_closing(mask, skmorph.disk(2))
    mask = ndimage.binary_fill_holes(mask)

    grid_quads, grid_tri = unwrap3D_meshtools.get_uv_grid_quad_connectivity(mask, 
                                                                            return_triangles=True, 
                                                                            bounds='none')
    grid_pts = np.dstack(np.indices(mask.shape)).reshape(-1,2)
    grid_pts = np.hstack([np.zeros(len(grid_pts))[:,None], 
                            grid_pts])

    grid_bool = mask.ravel() > 0
    face_bool = grid_bool[grid_tri].copy()
    face_invalid_index = np.unique(np.argwhere(face_bool==0)[:,0])
    face_keep_index = np.setdiff1d(np.arange(len(grid_tri)), face_invalid_index)

    mesh_2D = unwrap3D_meshtools.create_mesh(grid_pts[:,:], grid_tri[face_keep_index][:,::-1])
    mesh_2D_comps = mesh_2D.split(only_watertight=False)
    mesh_2D_submesh = mesh_2D_comps[np.argmax([len(ccc.vertices) for ccc in mesh_2D_comps])]

    v = mesh_2D_submesh.vertices
    f = mesh_2D_submesh.faces

    bdy_index = igl.boundary_loop(f) # Here returns the index in (v) vertices
    boundary_point = v[bdy_index][:,1:]

    return boundary_point, mesh_2D_submesh, bdy_index

def extract_igl_boundary_stack(img_stack):
    """
    Perform stack operation to generate igl boundaries
    
    """
    boundary_stack = []
    mesh_2D_submesh_stack = []
    bdy_index_stack = []
    for ii, img in enumerate(img_stack):

        boundary_points, mesh_2D_submesh, bdy_index = extract_igl_boudnary(img) 
    
        boundary_stack.append(boundary_points)
        mesh_2D_submesh_stack.append(mesh_2D_submesh)
        bdy_index_stack.append(bdy_index)
    return boundary_stack, mesh_2D_submesh_stack, bdy_index_stack

"""
##########################################################################################################################
Below are the functions for propagating displacement field to boundaries

"""


def normalize_gradient_vector_field(mask_dist_tform_gradients):
    """
    Normalized 
    
    """
    length = np.linalg.norm(mask_dist_tform_gradients, axis=-1, keepdims=True)
    epsilon = 1e-20
    normalized_gvf = mask_dist_tform_gradients / (length + epsilon)
    return normalized_gvf


def generate_point_mapping_gradient(mask_outside):

    """
    This function is generating the gradient for mapping the warped points onto the boundary

    """
    import unwrap3D.Segmentation.segmentation as unwrap3D_segmentation
    from copy import deepcopy
    ### This is the gradient from outside to inside 
    mask_dist_tform_outside = unwrap3D_segmentation.sdf_distance_transform(mask_outside)    
    mask_dist_tform_gradients_outside = np.array(np.gradient(mask_dist_tform_outside)).transpose(1,2,0)

    ### This is the gradient from inside to outside 
    mask_inside = ~mask_outside
    mask_dist_tform_inside = unwrap3D_segmentation.sdf_distance_transform(mask_inside)
    mask_dist_tform_gradients_inside = np.array(np.gradient(mask_dist_tform_inside)).transpose(1,2,0)

    mask_dist_tform_gradients = deepcopy(mask_dist_tform_gradients_outside)
    mask_dist_tform_gradients[mask_outside>0] = mask_dist_tform_gradients_inside[mask_outside>0]

    return mask_dist_tform_gradients



def propagate_dxdy_forward_gradient(boundary_points, displacement_field, contour, mask, alpha = 1, max_iter = 50,  output_dir=None):

    """
    Method 1: use x' = x + dx; y' = y + dy 
    Forward euler
    Return before_map for debug purpose
    
    """
    """Propagate boundary points using a displacement field and map back to contour."""
    import unwrap2D as unwrap2D_fns
    import os
    from scipy.ndimage import map_coordinates
    mask_dist_tform_gradients = generate_point_mapping_gradient(mask)
    normalized_gvf = normalize_gradient_vector_field(mask_dist_tform_gradients)
    dx = displacement_field[..., 0]  
    dy = displacement_field[..., 1]  


    before_map = []
    displaced_boundary = []
    

    for ii, (y, x) in enumerate(boundary_points):  
        dx_interp = map_coordinates(dx, [[y], [x]], order=1)[0]  
        dy_interp = map_coordinates(dy, [[y], [x]], order=1)[0]

        x_new = x + dx_interp
        y_new = y + dy_interp

        new_point = (y_new, x_new)
        before_map.append(new_point)
      
    last_points = np.array(before_map)
    distances = []
    points = []
    
    for ii in range(max_iter):
       

        contour_evolve = unwrap2D_fns.parametric_line_flow_2D(last_points,
                                                    external_img_gradient=normalized_gvf, 
                                                    E=None, 
                                                    close_contour=True, 
                                                    fixed_boundary = False, 
                                                    lambda_flow = alpha, # adjusts the balance.  (decrease this to be more similar to original cell shape ), increase to be more like version 1
                                                    step_size = 1, # adjusts the spacing between curves. 
                                                    niters = 1, 
                                                    conformalize = False,
                                                    eps = 1e-12)
                                                
        new_points = contour_evolve[:,:,1]
        
        chamfer_dist = chamfer_distance(new_points, contour)

        distances.append(chamfer_dist)
        points.append(new_points)
    
        last_points = new_points
    
    
        if output_dir:
            plt.figure(figsize=(5, 5))
            plt.imshow(mask, cmap='gray')
            plt.plot(contour[:, 1], contour[:, 0], 'g--', label='Original contour')
            plt.plot(new_points[:, 1], new_points[:, 0], 'r-', label=f'Iter {ii+1}')
            plt.legend()
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'frame_{ii:03d}.png'))
            plt.close()

    return last_points, np.array(before_map),  distances, points

def propagate_dxdy_forward_gradient_diffusionGVF(boundary_points, 
                                                 displacement_field, 
                                                 contour, 
                                                 mask, 
                                                 alpha = 1, 
                                                 w = 1,  
                                                 max_iter = 50,  
                                                 output_dir=None, 
                                                 GVF_diffusion=True):

    """
    Method 1: use x' = x + dx; y' = y + dy 
    Forward euler
    Return before_map for debug purpose
    
    """
    """Propagate boundary points using a displacement field and map back to contour."""
    import unwrap2D as unwrap2D_fns
    import os
    from scipy.ndimage import map_coordinates
    mask_dist_tform_gradients = generate_point_mapping_gradient(mask)
    normalized_gvf = normalize_gradient_vector_field(mask_dist_tform_gradients)

    diffusion_gvf = generate_point_mapping_gradient_diffusion(mask)
    diffusion_gvf = np.transpose(diffusion_gvf, (1, 2, 0))

    dx = displacement_field[..., 0]  
    dy = displacement_field[..., 1]  


    before_map = []
    displaced_boundary = []
    

    for ii, (y, x) in enumerate(boundary_points):  
        dx_interp = map_coordinates(dx, [[y], [x]], order=1)[0]  
        dy_interp = map_coordinates(dy, [[y], [x]], order=1)[0]
        # y = int(y)
        # x = int(x)
        # dx_interp = dx[y, x]
        # dy_interp = dy[y, x]

        x_new = x + w * dx_interp
        y_new = y + w * dy_interp

        new_point = (y_new, x_new)
        before_map.append(new_point)
        #mapped_point = map_point_automatic_stop(new_point, mask_dist_tform_gradients, contour)
       
        #displaced_boundary.append(mapped_point)
    last_points = np.array(before_map)
    distances = []
    points = []
    
    for ii in range(max_iter):
       

        if GVF_diffusion == True:
            
            contour_evolve = unwrap2D_fns.parametric_line_flow_2D(last_points,
                                                    external_img_gradient=diffusion_gvf, 
                                                    E=None, 
                                                    close_contour=True, 
                                                    fixed_boundary = False, 
                                                    lambda_flow = alpha, # adjusts the balance.  (decrease this to be more similar to original cell shape ), increase to be more like version 1
                                                    step_size = 1, # adjusts the spacing between curves. 
                                                    niters = 1, 
                                                    conformalize = False,
                                                    eps = 1e-12)
        else:
            contour_evolve = unwrap2D_fns.parametric_line_flow_2D(last_points,
                                                    external_img_gradient=normalized_gvf, 
                                                    E=None, 
                                                    close_contour=True, 
                                                    fixed_boundary = False, 
                                                    lambda_flow = alpha, # adjusts the balance.  (decrease this to be more similar to original cell shape ), increase to be more like version 1
                                                    step_size = 1, # adjusts the spacing between curves. 
                                                    niters = 1, 
                                                    conformalize = False,
                                                    eps = 1e-12)
                                                
        new_points = contour_evolve[:,:,1]
        
        # new_points = update_points_by_gradient(last_points, normalized_gvf, alpha = alpha)
        chamfer_dist = chamfer_distance(new_points, contour)

        distances.append(chamfer_dist)
        points.append(new_points)
    
        last_points = new_points
    
    
        if output_dir:
            plt.figure(figsize=(5, 5))
            plt.imshow(mask, cmap='gray')
            plt.plot(contour[:, 1], contour[:, 0], 'g--', label='Original contour')
            plt.plot(new_points[:, 1], new_points[:, 0], 'r-', label=f'Iter {ii+1}')
            plt.legend()
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'frame_{ii:03d}.png'))
            plt.close()


    return last_points, np.array(before_map),  distances, points


def generate_point_mapping_gradient_diffusion(mask_outside):

    """
    This function is generating the gradient for mapping the warped points onto the boundary

    """
    import unwrap3D.Segmentation.segmentation as unwrap3D_segmentation
    from copy import deepcopy
    ### This is the gradient from outside to inside 
    mask_dist_tform_outside = unwrap3D_segmentation.sdf_distance_transform(mask_outside)    
    mask_dist_tform_gradients_outside = np.array(np.gradient(mask_dist_tform_outside)).transpose(1,2,0)
    gvf_outside = GVF_diffuse2D(mask_dist_tform_gradients_outside, normalize=True)
    
    mask_inside = 1-mask_outside

    mask_dist_tform_inside = unwrap3D_segmentation.sdf_distance_transform(mask_inside)
    mask_dist_tform_gradients_inside = np.array(np.gradient(mask_dist_tform_inside)).transpose(1,2,0)
    gvf_inside = GVF_diffuse2D(mask_dist_tform_gradients_inside, normalize=True)

    mask_dist_tform_gradients = deepcopy(mask_dist_tform_gradients_outside)
    mask_dist_tform_gradients[mask_outside>0] = mask_dist_tform_gradients_inside[mask_outside>0]
  
    gvf_combined = deepcopy(gvf_inside)

    ys, xs = np.nonzero(mask_inside)   

    gvf_combined[:, ys, xs] = gvf_outside[:, ys, xs]

    return gvf_combined

"""
##########################################################################################################################

Below are the functions for detect anomalies in the boundary propagation and restart function


"""

def cal_contour_distance(updated_boundary, contour):
    from scipy.spatial import KDTree
    tree = KDTree(contour)
    dists, _ = tree.query(updated_boundary)
    return dists

def cal_section_displacement(points_t, points_tp1):
    displacement = []
    for st, st1 in zip(points_t, points_tp1):
    
        center = np.mean(st, axis = 1)
        center1 = np.mean(st1, axis = 1)
        displacement.append(np.linalg.norm(center-center1))

    return displacement

def length_proportion(points, section = 0.1):
    import math
    sections = partition_points(points,section)
   
    portion_list = []
    # calculate the arc length
    for pp in sections:
        arc_length = cal_arc_length(pp)
        portion_list.append(arc_length)
    

    length_sum = np.sum(portion_list)
    portion_list = portion_list/length_sum
   
    return portion_list, sections

def partition_points(points,section=0.1):
    import math
    sections = []

    n_section = math.ceil(1 / section)
    n_points = len(points)

    for ii in range(n_section):
        if ii != (n_section - 1):
            s = int(n_points * section * ii)
            e = int(n_points * section * (ii + 1))
        else:
            s = int(n_points * section * ii)
            e = int(n_points)

        section_point = points[s:e]
        sections.append(section_point)

    return sections



def chamfer_distance(point_set1, point_set2):
    """
    Compute the Chamfer Distance between two point sets.

    """
    from scipy.spatial import cKDTree
    import numpy as np

    tree1 = cKDTree(point_set1)
    tree2 = cKDTree(point_set2)

    dist1, _ = tree2.query(point_set1)
    dist2, _ = tree1.query(point_set2)

    chamfer_dist = np.mean(dist1**2) + np.mean(dist2**2)
    return chamfer_dist











"""
##########################################################################################################################
Below are the functions for reordering the boundary points for both warped_boundaries and original
igl_boundary
for warped_boundaries, it needs to be reordered because the kdtree mapping will cause intersection
(could be potentially solved by cMCF)

for igl_boundaries, it needs to be reordered because the first point of the boundaries might be changing

"""


def reorder_warped_boundary(igl_boundary, warped_boundary):
    """
    This method use the order of igl_boundary to reorder the warped_boundary;
    In this way the warped_boundary can remove intersection

    And always fixed the first point
     
    This function can be discarded because the new boundary mapping methods guaranteen the order
    """
    reordered_boundary_second_half = []
    reordered_boundary_first_half = []
    first_half = False
    
    for point in igl_boundary:
        #index = np.where(np.all(warped_boundary == point, axis=1))[0] # here needs to be changed to find the closest points
        dists = np.linalg.norm(warped_boundary - point, axis=1)
        ii = np.argmin(dists)
        print(ii)
        # for ii in index:
        
        if ii == 0: 
            # Put the first point in the warped_boundary to the first point
            first_half = True

        if first_half is False:
            reordered_boundary_second_half.append(warped_boundary[ii])
        else:
            reordered_boundary_first_half.append(warped_boundary[ii])


    if len(reordered_boundary_second_half) == 0:
        reordered_boundary = np.array(reordered_boundary_first_half)
    else:
        reordered_boundary_first_half = np.array(reordered_boundary_first_half).reshape(-1, 2)
        reordered_boundary_second_half = np.array(reordered_boundary_second_half).reshape(-1, 2)
        
        reordered_boundary = np.concatenate((reordered_boundary_first_half, reordered_boundary_second_half),axis = 0)
    return reordered_boundary

def reorder_warped_boundary_stack(igl_boundary_stack, warped_boundary_stack):
    reorder_boundaries = []

    for igl_boundary, warped_boundary in tqdm(zip(igl_boundary_stack, warped_boundary_stack)):
       
        reorder_boundaries.append(reorder_warped_boundary(igl_boundary, warped_boundary))

    return reorder_boundaries


def reorder_igl_boundary(first_point, igl_boundary):
    """
    This function use the first point in the first frame to align all the frame igl_boundary
    """
    # idx = np.where(np.all(igl_boundary == first_point, axis = 1))[0][0]
    dists = np.linalg.norm(igl_boundary - first_point, axis=1)
    idx = np.argmin(dists)
    reorder_boundary = np.concatenate((igl_boundary[idx:], igl_boundary[:idx]),axis=0)
    return reorder_boundary

def reorder_igl_boundary_stack(igl_unwarped_boundary_stack, igl_reordered_warped_boundary_stack):
    """
    use the first point to align the igl boundaries for stack operation
    """
    igl_reordered_unwarped = []
    for igl_unwraped, igl_reordered in zip(igl_unwarped_boundary_stack, igl_reordered_warped_boundary_stack):
        first_point = igl_reordered[0]
        igl_reordered_unwarped.append(reorder_igl_boundary(first_point, igl_unwraped))
    return igl_reordered_unwarped

def _resample_curve(x,y,k=1, s=0, n_samples=10):
    
    import scipy.interpolate
     
    tck, u = scipy.interpolate.splprep([x,y], k=k, s=s)
    unew = np.linspace(0, 1.00, n_samples)
    out = scipy.interpolate.splev(unew, tck) 
    
    return np.vstack(out).T

def detect_anomalies(updated_boundary, reference_boundary, global_dists, threshold = 2.5):
    """
    Functions to detect failed boundary propagation    
    
    """
    from scipy.spatial import KDTree
    # n_samples = len(updated_boundary)
    # reference_resampled = _resample_curve(reference_boundary[:,0],reference_boundary[:,1],k=1, s=0, n_samples=n_samples) 
    

    tree = KDTree(reference_boundary)
    dists, _ = tree.query(updated_boundary)
    
    if len(global_dists)>0:
        dist_mean = np.mean(global_dists)
        dist_std = np.std(global_dists)
    else:
        dist_mean = np.mean(dists)
        dist_std = np.std(dists)
    print(dist_mean)
    print(dist_std)
    
    anomalies = np.where(
        (dists > threshold)
    )[0]
    normal_dists = np.delete(dists, anomalies)
    global_dists.extend(normal_dists)
    
    has_anomalies = len(anomalies) > 0


    return has_anomalies, anomalies, dists, global_dists



def boundary_propagation_with_anomaly_detection(
                                                displacement_fields_stack, 
                                                mask_stack, 
                                                igl_unwarped_boundary_stack, 
                                                output_dir,
                                                window_size = 6,      
                                                sigma_thresh = 3.0,  
                                                min_history = 5,       
                                                contour_distance_thres = 3.0,
                                                stats_plot = True):

        from scipy.spatial import cKDTree
        from collections import deque
        last_frame_boundary = igl_unwarped_boundary_stack[0]
        boundary_points_stack_euler = [last_frame_boundary]
        before_map_stack_euler = [last_frame_boundary]
        pure_resample = [last_frame_boundary]
        n_samples = len(igl_unwarped_boundary_stack[0])



        dq_disp_mean = deque(maxlen=window_size)    
        dq_portion_var = deque(maxlen=window_size)   
        dq_ct_dis = deque(maxlen=window_size) 

        timeseries_disp = []
        timeseries_portion = []
        timeseries_ct_dis = []
        ts_ct_dis = []

        os.makedirs(os.path.join(output_dir, "boundary_visualization"), exist_ok=True)
        for ii in tqdm(range(1, len(mask_stack))):
            #img = registered_img_stack[ii]
            mask = mask_stack[ii].astype(np.bool_)
            contour = igl_unwarped_boundary_stack[ii]
            
            GVF_vis_dir = os.path.join(output_dir, f"frame_{ii}")
            os.makedirs(GVF_vis_dir, exist_ok=True)
            displacement = displacement_fields_stack[ii-1]
            warped_boundary, before_map, _, _ = propagate_dxdy_forward_gradient_diffusionGVF(
                last_frame_boundary, displacement, contour, mask,
                alpha=1, max_iter=100,
                output_dir=GVF_vis_dir
            )
            boundary_points_stack_euler.append(warped_boundary)
            before_map_stack_euler.append(before_map)

            if output_dir is not None:
                plt.plot(warped_boundary[:,1], warped_boundary[:,0], 'ro-', linewidth=0.5, markersize=0.5, label='warped whole')
                plt.legend()
                plt.savefig(os.path.join(output_dir, "boundary_visualization",f"frame_{ii:02d}.png"))
                plt.close()

            if ii == 1:
                portion_list, sections = length_proportion(last_frame_boundary, section=0.1)
                p_var = np.var(portion_list * 100)
            portion_list2, sections2 = length_proportion(warped_boundary, section=0.1)
            p_var = np.var(portion_list2 * 100) 

            timeseries_portion.append(p_var)
            dq_portion_var.append(p_var)

            displacement_vals = cal_section_displacement(sections, sections2)  # list of displacements per section
            disp_mean_curr = float(np.mean(displacement_vals))
            disp_std_curr = float(np.std(displacement_vals))
            timeseries_disp.append(disp_mean_curr)
            dq_disp_mean.append(disp_mean_curr)

            contour_distance = cal_contour_distance(warped_boundary, contour)
            if np.any(contour_distance) > contour_distance_thres:
                ctds_anom = True
            else:
                ctds_anom = False
            # mean = np.mean(contour_distance)
            # std = np.std(contour_distance)
            # ratio = np.max(contour_distance)/np.median(contour_distance)

            # if ratio > 5:
            #     ctds_anom = True
            # else:
            #     ctds_anom = False


            ct_dis_mean_curr = float(np.mean(contour_distance))
            ct_dis_std_curr = float(np.std(contour_distance))
            timeseries_ct_dis.append(ct_dis_mean_curr)
            dq_ct_dis.append(ct_dis_mean_curr)
            ts_ct_dis.append(ct_dis_mean_curr)

            if len(dq_disp_mean) >= min_history:
                
                hist_disp = np.array(list(timeseries_disp)[:-1]) if len(timeseries_disp) > 1 else np.array(list(timeseries_disp))
                hist_port = np.array(list(timeseries_portion)[:-1]) if len(timeseries_portion) > 1 else np.array(list(timeseries_portion))
                hist_ctds = np.array(list(ts_ct_dis)[:-1]) if len(ts_ct_dis) > 1 else np.array(list(ts_ct_dis))

                disp_mean_hist = hist_disp.mean()
                disp_std_hist = hist_disp.std() if hist_disp.size > 1 else 0.0
                port_mean_hist = hist_port.mean()
                port_std_hist = hist_port.std() if hist_port.size > 1 else 0.0
                ct_dist_mean_hist = hist_ctds.mean()
                ct_dist_std_hist = hist_ctds.std() if hist_ctds.size > 1 else 0.0
                
                if disp_std_hist == 0:
                    disp_std_hist = 1e-8
                if port_std_hist == 0:
                    port_std_hist = 1e-8
                if ct_dist_std_hist == 0:
                    ct_dist_std_hist = 1e-8

            
                displacement_anom = (disp_mean_curr > disp_mean_hist + sigma_thresh * disp_std_hist) 
                portion_anom = (p_var > port_mean_hist + sigma_thresh * port_std_hist)
                #ctds_anom = (ct_dis_mean_curr > ct_dist_mean_hist + sigma_thresh * ct_dist_std_hist)
                
                if displacement_anom or portion_anom or ((ct_dis_mean_curr>contour_distance_thres) and ctds_anom):
                    
                    first_point = warped_boundary[0]

                    print("==== Anomaly detected ====")
                    print(f"frame: {ii}, disp_mean_curr: {disp_mean_curr:.4f}, disp_hist_mean: {disp_mean_hist:.4f}, disp_hist_std: {disp_std_hist:.4f}, ctds_anom:{ctds_anom}")
                    print(f"portion_curr: {p_var:.4f}, port_hist_mean: {port_mean_hist:.4f}, port_hist_std: {port_std_hist:.4f}")
                    print(f"contour distance: {ct_dis_mean_curr:.4f}, ct_dist_mean_hist: {ct_dist_mean_hist:.4f}, ct_dist_std_hist: {ct_dist_std_hist:.4f}")
                    print("==========================")

            
                    plt.figure(figsize=(6,6))
                    plt.imshow(mask, cmap='gray')
                    reference_resampled = _resample_curve(contour[:,0], contour[:,1], k=1, s=0, n_samples=n_samples)
                    plt.plot(warped_boundary[:200,1], warped_boundary[:200,0], 'ro-', linewidth=0.5, markersize=0.5, label='warped (first 200)')
                    plt.plot(reference_resampled[:200,1], reference_resampled[:200,0], 'go-', linewidth=0.5, markersize=0.5, alpha=0.7, label='reference (first 200)')
                    plt.legend()
                    plt.title(f"Anomaly at frame {ii}")
                    plt.show()

                    tree = cKDTree(reference_resampled)
                    dist, idx = tree.query(first_point)
                    warped_boundary =  np.vstack([
                        reference_resampled[idx:],   
                        reference_resampled[:idx]    
                    ])

            
                else:
                    warped_boundary = _resample_curve(warped_boundary[:,0], warped_boundary[:,1], k=1, s=0, n_samples=n_samples)
            
            reference_resampled = _resample_curve(contour[:,0],contour[:,1],k=1, s=0, n_samples=n_samples)
            fp = pure_resample[-1][0]
            tree = cKDTree(reference_resampled)
            dist, idx = tree.query(fp)
            resample_boundary =  np.vstack([
                reference_resampled[idx:],   
                reference_resampled[:idx]    
            ])
            pure_resample.append(resample_boundary)


            portion_list = portion_list2
            sections = sections2
            last_frame_boundary = warped_boundary
            
        if stats_plot:
            fig, ax = plt.subplots(3,1, figsize=(10,6), sharex=True)
            t = np.arange(1, 1+len(timeseries_disp))
            ax[0].plot(t, timeseries_disp, label='disp_mean')
            ax[0].set_title('displacement mean time series')
            ax[1].plot(t, timeseries_portion, label='portion_var')
            ax[1].set_title('portion var time series')
            ax[2].plot(t, timeseries_ct_dis,label='contour_distance_mean')
            ax[2].set_title('distance from contour')
            plt.tight_layout()
            plt.show()
        return boundary_points_stack_euler, pure_resample

def _EnforceMirrorBoundary2D(f):

    if f.ndim != 2:
        raise ValueError("_EnforceMirrorBoundary2D expects a 2D array")

    N, M = f.shape

    if N < 5 or M < 5:
        return f

    xi = np.arange(1, M-2)
    yi = np.arange(1, N-2)

    f[0, 0] = f[2, 2]
    f[0, M-1] = f[2, M-3]
    f[N-1, 0] = f[N-3, 2]
    f[N-1, M-1] = f[N-3, M-3]

    if xi.size > 0:
        f[np.ix_([0, N-1], xi)] = f[np.ix_([2, N-3], xi)]

    if yi.size > 0:
        f[np.ix_(yi, [0, M-1])] = f[np.ix_(yi, [2, M-3])]

    return f


def GVF_diffuse2D(vector_field, mu=0.01, iterations=50, normalize=True):
    from scipy.ndimage import laplace as del2
    vf = np.array(vector_field, copy=True)
    if vf.ndim == 3 and vf.shape[2] == 2:

        vf = np.transpose(vf, (2, 0, 1))
    elif vf.ndim == 3 and vf.shape[0] == 2:
        pass
    else:
        raise ValueError("vector_field must have shape (2, H, W) or (H, W, 2)")

    if normalize:
        norms = np.linalg.norm(vf, axis=0)
        vf = vf / (norms[None, ...] + 1e-20)
    else:
        vf = vf.copy()

    Fx, Fy = vf 
    magSquared = Fx*Fx + Fy*Fy

    u = Fx.copy()
    v = Fy.copy()

    H, W = Fx.shape

    for i in range(iterations):

        u = _EnforceMirrorBoundary2D(u)
        v = _EnforceMirrorBoundary2D(v)


        u = u + mu * 4 * del2(u) - (u - Fx) * magSquared
        v = v + mu * 4 * del2(v) - (v - Fy) * magSquared

        if normalize:
            mag = np.sqrt(u**2 + v**2)
            u = u / (mag + 1e-12)
            v = v / (mag + 1e-12)

        Fx = u.copy()
        Fy = v.copy()
        magSquared = Fx*Fx + Fy*Fy


    vector_field_new = np.array([u, v])
    return vector_field_new


def cal_arc_length(points, loop = False):
    """
    Calculate the arc length give a list of points
    
    """
    if loop:
        points.append(points[0]) 

    dist = points[1:] - points[:-1]
    length = np.sum(np.linalg.norm(dist,axis=0))
    
 
    return length


"""
Code for:
    Detect boundary disorder and fix it by interpolation

"""
def find_disordered_runs(ref_idx, low_thresh=0, high_thresh=5.0, min_run_len=4):
    
    N = len(ref_idx)
    ref_idx = ref_idx - ref_idx[0]
    if N < 3:
        return [], np.zeros(0, dtype=bool)
    
    raw_diff = np.diff(ref_idx) 
    diff = (raw_diff + N // 2) % N - N // 2
    # print(ref_idx)
    # print(diff)
    bad_mask = (diff < low_thresh) | (diff > high_thresh)

    runs = []
    i = 0
    while i < len(bad_mask):
        if not bad_mask[i]:
            i += 1
            continue
        start = i
        while i < len(bad_mask) and bad_mask[i]:
            i += 1
        end = i  
        if (end - start + 1) >= min_run_len:
            runs.append((start, end))
    return runs, bad_mask

  

  

def extract_shortest_arc(contour, i_start, i_end):

    N = len(contour)
    i_start = int(i_start) % N
    i_end   = int(i_end)   % N

    if i_end >= i_start:
        forward_idx = np.arange(i_start, i_end + 1)
    else:
        forward_idx = np.concatenate([
            np.arange(i_start, N),
            np.arange(0, i_end + 1)
        ])
    if i_start >= i_end:
        backward_idx = np.arange(i_start, i_end - 1, -1)
    else:
        backward_idx = np.concatenate([
            np.arange(i_start, -1, -1),
            np.arange(N - 1, i_end - 1, -1)
        ])

    if len(forward_idx) <= len(backward_idx):
        idx_use = forward_idx
    else:
        idx_use = backward_idx

    return contour[idx_use]

def fix_warped_segment_by_ref_interp(ref_boundary,
                                     warped_boundary,
                                     ref_idx,
                                     runs):

    corrected = deepcopy(warped_boundary)
    tree_ref = cKDTree(ref_boundary)
    N_ref = len(ref_boundary)
    N = len(warped_boundary)
    for (s, e) in runs:
        s = s %  N # s and e is already the "bad point"
        e = e % N
        L = e - s + 1
        if L < 2:
            continue

        P_s = warped_boundary[s]
        P_e = warped_boundary[e]

        _, idx_s = tree_ref.query(P_s)
        _, idx_e = tree_ref.query(P_e)

        arc = extract_shortest_arc(ref_boundary, idx_s, idx_e)
        if len(arc) < 2:
            continue

        arc_resampled = _resample_polyline(arc, L)

   
        if np.linalg.norm(arc_resampled[0] - P_s) > np.linalg.norm(arc_resampled[-1] - P_s):
            arc_resampled = arc_resampled[::-1]

        corrected[s:e+1] = arc_resampled

    return corrected

def _resample_polyline(points, n_samples):
   
    pts = np.asarray(points, dtype=float)
    M = len(pts)
    if M == 0:
        return np.zeros((n_samples, 2), dtype=float)
    if M == 1:
        return np.repeat(pts, n_samples, axis=0)
    if M == n_samples:
        return pts.copy()

    diffs = pts[1:] - pts[:-1]         
    seglen = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglen)])  
    total_len = s[-1]

    if total_len == 0:

        return np.repeat(pts[:1], n_samples, axis=0)

    s_norm = s / total_len               
    t_new = np.linspace(0.0, 1.0, n_samples)

    y_new = np.interp(t_new, s_norm, pts[:, 0])
    x_new = np.interp(t_new, s_norm, pts[:, 1])
    return np.stack([y_new, x_new], axis=1)

def get_ref_indices_from_warped(ref_boundary, warped_boundary):
        tree = cKDTree(ref_boundary)
        dists, idxs = tree.query(warped_boundary)
    
        return idxs


def align_boundary_to_first_point(boundary, first_point):
   
    tree = cKDTree(boundary)
    dist, idx = tree.query(first_point)
    aligned = np.vstack([boundary[idx:], boundary[:idx]])
    return aligned, idx

def det_run_max_length(runs):
    max_run = 0
    for run in runs:
        dif = np.abs(run[1]-run[0])
        max_run = max(max_run, dif)
    return max_run

def det_run_all_length(runs):
    all_run = 0
    for run in runs:
        dif = np.abs(run[1]-run[0])
        all_run +=  dif
    return all_run

def merge_runs(runs):
    runs_sorted = sorted(runs, key=lambda x: x[0])

    merged = []
    cur_start, cur_end = runs_sorted[0]

    for start, end in runs_sorted[1:]:
        if start <= cur_end + 1:

            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    merged.append((cur_start, cur_end))
    return merged


def boundary_propagation_with_anomaly_detection_fix_boundary(
                                                displacement_fields_stack, 
                                                mask_stack, 
                                                igl_unwarped_boundary_stack, 
                                                output_dir,
                                                window_size = 6,      
                                                sigma_thresh = 3.0,  
                                                min_history = 5,       
                                                contour_distance_thres = 3.0,
                                                alpha = 5,
                                                stats_plot = True):
    """
    Function similary to boundary_propagation_with_anomaly_detection, but perform a detection of points disorder and fix it by interpolation
    """
    from collections import deque
    
    from meshlib import mrmeshpy, mrmeshnumpy



    last_frame_boundary = igl_unwarped_boundary_stack[0]
    boundary_points_stack_euler = [last_frame_boundary]
    before_map_stack_euler = [last_frame_boundary]
    pure_resample = [last_frame_boundary]

    n_samples = len(igl_unwarped_boundary_stack[0])

    dq_disp_mean = deque(maxlen=window_size)
    dq_portion_var = deque(maxlen=window_size)
    dq_ct_dis = deque(maxlen=window_size)

    timeseries_disp = []
    timeseries_portion = []
    timeseries_ct_dis = []
    ts_ct_dis = []

    os.makedirs(os.path.join(output_dir, "boundary_visualization"), exist_ok=True)

    for ii in tqdm(range(1, len(mask_stack))):
        #img = registered_img_stack[ii]
        mask = mask_stack[ii].astype(np.bool_)
        contour_raw = deepcopy(igl_unwarped_boundary_stack[ii])  
        contour = _resample_curve(contour_raw[:,0],contour_raw[:,1],k=1, s=0, n_samples=n_samples) 
    
        first_point = boundary_points_stack_euler[-1][0] 

        ref_boundary_aligned, shift_idx =  align_boundary_to_first_point(
            contour,
            first_point
        )

        reference_resampled = _resample_curve(contour[:,0],contour[:,1],k=1, s=0, n_samples=n_samples)
        fp = pure_resample[-1][0]
        tree = cKDTree(reference_resampled)
        dist, idx = tree.query(fp)
        resample_boundary =  np.vstack([
            reference_resampled[idx:],   
            reference_resampled[:idx]    
        ])
        pure_resample.append(resample_boundary)


        GVF_vis_dir = os.path.join(output_dir, f"frame_{ii}")
        os.makedirs(GVF_vis_dir, exist_ok=True)

        displacement = displacement_fields_stack[ii - 1]


        warped_boundary, before_map, _, _ = propagate_dxdy_forward_gradient_diffusionGVF(
            last_frame_boundary,
            displacement,
            ref_boundary_aligned, 
            mask,
            alpha=alpha,
            max_iter=100,
            output_dir=GVF_vis_dir
        )
        

        warped_before_fix = warped_boundary.copy()

        polyline = mrmeshnumpy.polyline2FromPoints(warped_before_fix)
        selfInters = mrmeshpy.findSelfCollidingEdges(polyline)
        pad = 5
        runs = []
   
        for inter in selfInters:
            i = int(inter.aUndirEdge)
            j = int(inter.bUndirEdge)

            runs.append((max(i - pad, 0),
                     min(i + 1 + pad, n_samples)))

            runs.append((max(j - pad, 0),
                     min(j + 1 + pad, n_samples)))
        if len(runs) > 0:
            runs = merge_runs(runs)
            
        if ii == 1:
            portion_list_prev, sections_prev = length_proportion(last_frame_boundary, section=0.1)
        portion_curr, sections_curr = length_proportion(warped_boundary, section=0.1)

        p_var = np.var(portion_curr * 100.0)
        timeseries_portion.append(p_var)
        dq_portion_var.append(p_var)

        displacement_vals = cal_section_displacement(sections_prev, sections_curr)
        disp_mean_curr = float(np.mean(displacement_vals))
        disp_std_curr = float(np.std(displacement_vals))
        timeseries_disp.append(disp_mean_curr)
        dq_disp_mean.append(disp_mean_curr)

        contour_distance = cal_contour_distance(warped_boundary, ref_boundary_aligned)
        ratio = np.max(contour_distance) / max(np.median(contour_distance), 1e-8)
        ctds_anom = bool(ratio > 5.0)

        ct_dis_mean_curr = float(np.mean(contour_distance))
        ct_dis_std_curr = float(np.std(contour_distance))
        timeseries_ct_dis.append(ct_dis_mean_curr)
        dq_ct_dis.append(ct_dis_mean_curr)
        ts_ct_dis.append(ct_dis_mean_curr)


        ref_idx = get_ref_indices_from_warped(ref_boundary_aligned, warped_boundary)
        
        # runs, dif = find_disordered_runs(
        #     ref_idx,    
        #     low_thresh=0,  
        #     high_thresh=5.0,
        #     min_run_len=4
        # )
        
        #n_runs = len(runs)
        runs_all_length = det_run_all_length(runs)

        if runs_all_length == 0:
            boundary_points_stack_euler.append(warped_boundary)

        elif runs_all_length <= 75:

            print(f"[frame {ii}] detected disordered runs (global order change): {runs}")
       
            warped_boundary = fix_warped_segment_by_ref_interp(
                ref_boundary=ref_boundary_aligned,
                warped_boundary=warped_boundary,
                ref_idx=ref_idx,
                runs=runs
            )
            boundary_points_stack_euler.append(warped_boundary)
            # plt.plot(warped_boundary[:, 1], warped_boundary[:, 0],
            #     'b--', linewidth=0.5, label='fixed')
        else:
            print(f"frame {ii} needs restart! Too many disordered runs: {runs}.")
            # print(f"dif:{dif}")
            warped_boundary = ref_boundary_aligned
            boundary_points_stack_euler.append(warped_boundary)

        warped_boundary = _resample_curve(
                warped_boundary[:, 0],  
                warped_boundary[:, 1],  
                k=1, s=0,
                n_samples=n_samples
            )

        plt.figure()
        plt.imshow(mask, cmap='gray')
        plt.plot(ref_boundary_aligned[:, 1], ref_boundary_aligned[:, 0],
                'b--', linewidth=0.5, label='ref aligned')
        plt.plot(warped_before_fix[:, 1], warped_before_fix[:, 0],
                'r.-', linewidth=0.5, markersize=1, label='warped before fix')
        plt.plot(warped_boundary[:, 1], warped_boundary[:, 0],
                'g.-', linewidth=0.5, markersize=1, label='warped after fix+resample')
        plt.legend()
        plt.title(f"Boundary propagation frame {ii}")
        plt.savefig(os.path.join(output_dir, "boundary_visualization", f"frame_{ii:02d}.png"))
        plt.close()


        portion_list_prev = portion_curr
        sections_prev = sections_curr
        last_frame_boundary = warped_boundary
        before_map_stack_euler.append(before_map)

    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    t = np.arange(1, 1 + len(timeseries_disp))
    ax[0].plot(t, timeseries_disp, label='disp_mean')
    ax[0].set_title('displacement mean time series')
    ax[1].plot(t, timeseries_portion, label='portion_var')
    ax[1].set_title('portion var time series')
    ax[2].plot(t, timeseries_ct_dis, label='contour_distance_mean')
    ax[2].set_title('distance from contour')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'track_stats.png'))
    plt.show()
    return boundary_points_stack_euler, pure_resample 


"""
2026.1.13 test
"""

def find_consecutive_runs(idx, min_length=4):
    idx = np.asarray(idx)
    if len(idx) == 0:
        return []

    runs = []
    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:

            if prev - start + 1 >= min_length:
                runs.append((start, prev))
            start = i
            prev = i


    if prev - start + 1 >= min_length:
        runs.append((start, prev))

    return runs

"""
#################################################################################################
Utils/Saving function function
For sanity check the warped boundary(connectivity, intersection and coverage of the images)

"""
def save_matplot_video(img, boundaries, save_path, save_name, img_plot = False,colormap='jet',scatter = False, plot_multiple_point = False):

    from PIL import Image
    tiff_frames = []
    vmin = np.percentile(img, 1)
    vmax = np.percentile(img, 99)

    for i, frame in enumerate(img):
        fig, ax = plt.subplots(figsize=(6, 6))
        if img_plot:
            im = ax.imshow(frame, cmap=colormap, vmin=vmin, vmax=vmax)
        # if scatter:
        #     ax.scatter(boundaries[i][:,1],boundaries[i][:,0],  c='red', s = 0.05,alpha=0.5)
        # else:
        #     ax.plot(boundaries[i][:,1],boundaries[i][:,0], 'ro-', linewidth = 0.05, alpha=0.5)

        if plot_multiple_point == True:
            """
            For check if the order of points are aligned
            """
            bdy_l = len(boundaries[i])
            color_list = [
                    "#1f77b4",  # blue
                    "#ff7f0e",  # orange
                    "#2ca02c",  # green
                    "#d62728",  # red
                    "#9467bd",  # purple
                    "#8c564b",  # brown
                    "#e377c2",  # pink
                    "#ffd700",  # gold 
                    "#17becf",  # cyan
                    "#bcbd22",  # olive
                
                ]
      
            for j in range(1,11):
                percentile_1 = int(bdy_l*(j-1)/10.)
                percentile_2 = int(bdy_l*j/10.)
                if j == 1:
                    #ax.plot(boundaries[i][percentile_1:percentile_2, 1], boundaries[i][percentile_1:percentile_2,0],marker = 'o', color = 'magenta', linewidth=0.5,markersize=0.5)
                    ax.scatter(boundaries[i][percentile_1:percentile_2, 1], boundaries[i][percentile_1:percentile_2,0],marker = 'o', color = 'magenta',s=1)
                elif (j < 10) and (j > 1):
                    #ax.plot(boundaries[i][percentile_1:percentile_2, 1], boundaries[i][percentile_1:percentile_2,0], marker = 'o', color = color_list[j-1], linewidth=0.5,markersize=0.5)
                    ax.scatter(boundaries[i][percentile_1:percentile_2, 1], boundaries[i][percentile_1:percentile_2,0], marker = 'o', color = color_list[j-1],s=1)
                else:
                    #ax.plot(boundaries[i][percentile_1:, 1], boundaries[i][percentile_1:,0], marker = 'o', color = color_list[j-1], linewidth=0.5,markersize=0.5)
                    ax.scatter(boundaries[i][percentile_1:, 1], boundaries[i][percentile_1:,0], marker = 'o', color = color_list[j-1],s=1)
            
            

        plt.axis('off') 

        temp_filename = f"frame_{i}.tiff"
        plt.savefig(temp_filename, format='tiff', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)  
        
        img = Image.open(temp_filename)
        tiff_frames.append(img)
        os.remove(temp_filename)
    # Save all frames as a multi-frame TIFF
    output_tiff = os.path.join(save_path, f"{save_name}.tif")
    tiff_frames[0].save(output_tiff, save_all=True, append_images=tiff_frames[1:])


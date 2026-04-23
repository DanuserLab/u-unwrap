import numpy as np
def generate_bdy_uv(mesh_2D_submesh):
    import igl 
    import unwrap3D.Mesh.meshtools as unwrap3D_meshtools
    import skimage.morphology as skmorph
    import scipy.ndimage as ndimage


    # mask = skmorph.binary_closing(mask, skmorph.disk(1))
    # mask = ndimage.binary_fill_holes(mask)

    # # 1. build a mesh from the binary 
    # grid_quads, grid_tri = unwrap3D_meshtools.get_uv_grid_quad_connectivity(mask>0, 
    #                                                                     return_triangles=True, 
    #                                                                     bounds='none')

    # grid_pts = np.dstack(np.indices(mask.shape)).reshape(-1,2)
    # grid_pts = np.hstack([np.zeros(len(grid_pts))[:,None], 
    #                     grid_pts])
    # grid_bool = mask.ravel() > 0


    # face_bool = grid_bool[grid_tri].copy()
    # face_invalid_index = np.unique(np.argwhere(face_bool==0)[:,0])
    # face_keep_index = np.setdiff1d(np.arange(len(grid_tri)), face_invalid_index)
    # # keep_tri_indices = np.hstack([iii for iii in np.arange(len(grid_tri)) if np.sum(np.intersect1d(grid_tri, invalid_pts))==0])

    # mesh_2D = unwrap3D_meshtools.create_mesh(grid_pts[:,:], grid_tri[face_keep_index][:,::-1])
    # mesh_2D_comps = mesh_2D.split(only_watertight=False)
    # mesh_2D_submesh = mesh_2D_comps[np.argmax([len(ccc.vertices) for ccc in mesh_2D_comps])]


    disk_coords, bdy_index = unwrap3D_meshtools.rectangular_conformal_map(mesh_2D_submesh.vertices,
                                                            mesh_2D_submesh.faces[:,:],
                                                            corner=None,  
                                                            random_state=0,
                                                            return_bdy_index = True)
    bdy_uv = disk_coords[bdy_index]
    return mesh_2D_submesh, disk_coords, bdy_uv, bdy_index


# def uv_interpolation_window(propagated_xy, original_xy, bdy_uv_0, window=100, eps=1e-8):
#     """
#     Matches original_xy to propagated_xy using sequential tracking.
#     Only allows local jumps to preserve order.
#     """
#     from scipy.spatial import cKDTree
#     import numpy as np

#     tree = cKDTree(propagated_xy)
#     n = len(propagated_xy)
#     interpolated_uv = []
#     bdy_uv_ = np.vstack([bdy_uv_0, bdy_uv_0[0]]) 
#     _, prev_idx = tree.query(original_xy[0], k=1)
#     for i, point in enumerate(original_xy):
       
#         candidate_indices = np.arange(prev_idx, prev_idx + window + 1) % n

#         candidate_points = propagated_xy[candidate_indices]

#         subtree = cKDTree(candidate_points)
#         _, match_idx = subtree.query(point, k=1)

#         # dists = np.linalg.norm(candidate_points - pt, axis=1)
#         # min_dist = np.argmin(dists)
#         # match_idx = candidate_indices[min_dist]

#         prev_idx = candidate_indices[match_idx]
        
#         idx1 = prev_idx
#         idx2 = (idx1 + 1) % n
    
#         p1, p2 = propagated_xy[idx1], propagated_xy[idx2]
#         d1 = np.linalg.norm(point - p1)
#         d2 = np.linalg.norm(point - p2)

#         u1, v1 = bdy_uv_[idx1]
#         u2, v2 = bdy_uv_[idx2]

#         theta1 = np.arctan2(v1, u1)
#         theta2 = np.arctan2(v2, u2)
      
#         if np.abs(theta2 - theta1) > np.pi:
#             if theta2 > theta1:
#                 theta2 -= 2 * np.pi
#             else:
#                 theta1 -= 2 * np.pi

#         if d1 == 0:
#             w1, w2 = 1, 0
#         elif d2 == 0:
#             w1, w2 = 0, 1
#         else:
#             w1, w2 = 1 / d1, 1 / d2

#         theta_p = (w1 * theta1 + w2 * theta2) / (w1 + w2)
#         up, vp = np.cos(theta_p), np.sin(theta_p)

#         interpolated_uv.append([up, vp])

#     return np.array(interpolated_uv)

# def uv_interpolate_ordered(propagated_xy, original_xy, bdy_uv):
#     from scipy.spatial import KDTree
#     import numpy as np

#     tree = KDTree(propagated_xy)
#     interpolated_uv = []
#     n = len(propagated_xy)

#     #bdy_uv_ = np.vstack([bdy_uv, bdy_uv[0]])  

#     _, last_idx = tree.query(original_xy[0], k=1)
#     #last_idx = prev_idx

#     for i, point in enumerate(original_xy):
#         _, idx1 = tree.query(point, k=1)
#         # print("first point")
#         # print(propagated_xy[idx1])
      
#         delta = (idx1 - last_idx) % n
#         if i > 0 and delta > n // 2: # if the jump is abnormal
#             idx1 = last_idx
#         else:
#             last_idx = idx1

#         idx2 = (idx1 + 1) % n
#         # idx1, idx2 = idx
#         p1, p2 = propagated_xy[idx1], propagated_xy[idx2]
#         d1 = np.linalg.norm(point - p1)
#         d2 = np.linalg.norm(point - p2)

#         u1, v1 = bdy_uv[idx1]
#         u2, v2 = bdy_uv[idx2]

#         theta1 = np.arctan2(v1, u1)
#         theta2 = np.arctan2(v2, u2)
      
#         if np.abs(theta2 - theta1) > np.pi:
#             if theta2 > theta1:
#                 theta2 -= 2 * np.pi
#             else:
#                 theta1 -= 2 * np.pi

#         if d1 == 0:
#             w1, w2 = 1, 0
#         elif d2 == 0:
#             w1, w2 = 0, 1
#         else:
#             w1, w2 = 1 / d1, 1 / d2

#         theta_p = (w1 * theta1 + w2 * theta2) / (w1 + w2)
#         up, vp = np.cos(theta_p), np.sin(theta_p)
#         up = up/(np.sqrt(up**2 + vp**2)+1e-20)
#         vp = vp/(np.sqrt(up**2 + vp**2)+1e-20)
#         interpolated_uv.append([up, vp])

#     return np.array(interpolated_uv)
"""
Shiqiu 20260114 test

"""

def uv_interpolate_ordered(propagated_xy, original_xy, bdy_uv):
    from scipy.spatial import KDTree
    import numpy as np

    tree = KDTree(propagated_xy)
    interpolated_uv = []
    n = len(propagated_xy)

    #bdy_uv_ = np.vstack([bdy_uv, bdy_uv[0]])  


    _, last_idx = tree.query(original_xy[0], k=1)

    for i, point in enumerate(original_xy):


        dists, idxs = tree.query(point, k=2)
        idx1, idx2 = idxs


        # def is_adjacent(a, b):
        #     return (b == (a + 1) % n) or (a == (b + 1) % n)

        # if is_adjacent(i1, i2):
        #     idx1, idx2 = i1, i2
        # else:
 
        #     idx1 = (last_idx + 1) % n
        #     idx2 = (idx1 + 1) % n


        if ((idx1 - last_idx) % n) > ((idx2 - last_idx) % n):
            idx1, idx2 = idx2, idx1
        
        last_idx = idx1
        # idx1, idx2 = idx
        p1, p2 = propagated_xy[idx1], propagated_xy[idx2]
        d1 = np.linalg.norm(point - p1)
        d2 = np.linalg.norm(point - p2)

        u1, v1 = bdy_uv[idx1]
        u2, v2 = bdy_uv[idx2]

        theta1 = np.arctan2(v1, u1)
        theta2 = np.arctan2(v2, u2)
      
        if np.abs(theta2 - theta1) > np.pi:
            if theta2 > theta1:
                theta2 -= 2 * np.pi
            else:
                theta1 -= 2 * np.pi

        if d1 == 0:
            w1, w2 = 1, 0
        elif d2 == 0:
            w1, w2 = 0, 1
        else:
            w1, w2 = 1 / d1, 1 / d2

        theta_p = (w1 * theta1 + w2 * theta2) / (w1 + w2)
        up, vp = np.cos(theta_p), np.sin(theta_p)
        up = up/(np.sqrt(up**2 + vp**2)+1e-20)
        vp = vp/(np.sqrt(up**2 + vp**2)+1e-20)
        interpolated_uv.append([up, vp])

    return np.array(interpolated_uv)

def interpolate_2nn_stack(unique_igl_warped_boundary_stack, igl_unwarped_boundary_stack, bdy_uv_0):
    bdy_uv_stack = []
    for propagated_xy, original_xy in zip(unique_igl_warped_boundary_stack, igl_unwarped_boundary_stack):
        interpolated_uv = uv_interpolate_ordered(propagated_xy, original_xy, bdy_uv_0)
        assert len(interpolated_uv) == len(original_xy)
        bdy_uv_stack.append(interpolated_uv)
    return bdy_uv_stack
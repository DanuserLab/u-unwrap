def calculate_corner(vertices, f):
    """
    This function calculates the corner boundary vertices to prevent frame shifting across different frames. 
    The four corners correspond to the vertices at 0, 90, 180, and -90 degrees relative to the cell's centroid.
    These corner points can then be used as input to the 'rectangular_conformal_map' function.
    
    """
    import igl

    bdy_index = igl.boundary_loop(f)
    
    xy_coord = vertices[:,1:]
    centroid = np.mean(xy_coord, axis=0)
 
    vertices = vertices[bdy_index]
    angles = np.arctan2(vertices[:,2] - centroid[1], vertices[:, 1] - centroid[0])

    target_angles = np.array([0, np.pi/2, np.pi, -np.pi/2])

    closest_vertices_index = []

    for target in target_angles:
        closest_index = np.argmin(np.abs(angles - target))
        closest_vertices_index.append(bdy_index[closest_index])
     
    return closest_vertices_index

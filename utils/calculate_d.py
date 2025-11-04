import numpy as np
from typing import List
import os


def calculate_sign(contour_evolve: np.ndarray) -> np.ndarray:
    """
    This function calculates the sign of the distance between the two adjacent contours.

    """
    from matplotlib.path import Path

    n_contours = contour_evolve.shape[-1]
    signs = []

    for ii in range(n_contours-1):
        contour_old = contour_evolve[...,ii]
        contour_new = contour_evolve[...,ii+1]

        path = Path(contour_old)
        inside = np.array(path.contains_points(contour_new))

        signs.append(np.where(inside, 1, -1))

    return np.array(signs).T

def accumulate_d(d, reference_n):
    """
    This function accumulates the distance between the two adjacent contours until the reference_n.

    """
    d_accumulated = np.sum(d[:,:reference_n], axis=1)
    return d_accumulated


def plot_d(raw_img: np.ndarray, 
           d: np.ndarray, 
           contour_evolve: np.ndarray,
           contour_index: List,
           savefolder: str,
           show_img:bool = True) -> None:
    """
    This function plots the distance between the two adjacent contours for sanity check
    
    Input:
        raw_img: the raw image
        d: the distance between the two adjacent contours(for all contours calculated)
        contour_evolve: several contours that are evolved from the original contour;
        contour_index: A list of the contour index to be plotted


    Output:
        None
    """
    import matplotlib.pyplot as plt
    contour_evolve = contour_evolve[...,contour_index]
    d = d[:,contour_index[:-1]] 

    # pre-calculate the dx_dy for the contour_evolve
    dx_dy = contour_evolve[...,1:] - contour_evolve[...,:-1]
    plt.figure(figsize=(10, 10))

    for j in range(len(contour_index)-1):
        contour_j = contour_evolve[...,j]
        plt.plot(contour_j[:, 1], contour_j[:, 0], lw=1.5, label=f'Contour {contour_index[j]}')
        for i in range(len(contour_j)):
            d_i = d[i,j]

            length = np.linalg.norm(dx_dy[i,:,j])
            dx = d_i * (dx_dy[i, 1, j]) / length   
            dy = d_i * (dx_dy[i, 0, j]) / length
            
            if d_i < 0:
                plt.arrow(contour_j[i, 1], contour_j[i, 0], 
                    -dx, -dy, fc='green', ec='green',
                    head_width=0.1, head_length=0.1)
                
            elif d_i >= 0:
            
                plt.arrow(contour_j[i, 1], contour_j[i, 0], 
                    dx, dy, fc='red', ec='red',
                    head_width=0.1, head_length=0.1)
                
    plt.plot(contour_evolve[...,-1][:, 1], contour_evolve[...,-1][:, 0], lw=1.5, label=f'Contour {contour_index[-1]}')
    plt.legend()
    #plt.gca().set_aspect('equal', adjustable='box')
    if show_img:
        plt.imshow(raw_img, cmap='gray')
    plt.savefig(os.path.join(savefolder, 'd_protrustion_retraction.png'))
    plt.show()



def calculate_d(contour_evolve, reference_n):
    """
    This function calculate the distance between the two adjacent contours.
    
    Inputs: 
        contour_evolve: several contours that are evolved from the original contour; 
                    candidate references
        reference_n: the index of contours to be used as reference (0 is the original contour)
    
    Outputs:
        d: the distance between the two adjacent contours
        ac_d: the accumulated distance between the two adjacent contours until the reference_n
    """
    # assert contour.shape[0] == contour_evolve.shape[0], \
    #     f"AssertionError: contour.shape[0] ({contour.shape[0]}) != contour_evolve.shape[0] ({contour_evolve.shape[0]})"
    
    value_d = np.linalg.norm(contour_evolve[...,1:] - contour_evolve[...,:-1], axis=1)
    sign_d = calculate_sign(contour_evolve)
    d = value_d * sign_d
    ac_d = accumulate_d(d, reference_n)
    return d, ac_d





if __name__=="__main__":
    
    import pylab as plt 
    from matplotlib import cm 
    import os 
    import skimage.io as skio 
    import skimage.segmentation as sksegmentation
    import skimage.morphology as skmorph

    import skimage.measure as skmeasure
    import scipy.ndimage as ndimage
    from scipy.ndimage import binary_fill_holes

    import unwrap3D.Mesh.meshtools as unwrap3D_meshtools
    import unwrap3D.Segmentation.segmentation as unwrap3D_segmentation
    import unwrap3D.Image_Functions.image as unwrap3D_image_fn
    
    import unwrap2D as unwrap2D_fns
    import skimage.draw as skdraw

    basefolder = '/Users/yushiqiu/Desktop/UTSW/rotation_project/Gaudenz/track_window'
    datafolder = os.path.join(basefolder,'data/Registration')
    savefolder = os.path.join(basefolder,'output/2024_08_22')

    mask = os.path.join(datafolder, 'MDACell5_Masks.tif')
    raw = os.path.join(datafolder, 'MDACell5_NoRegistration.tif')

    masks = skio.imread(mask)
    raw = skio.imread(raw)

    test_frame = 51 
    # test_frame = 50

    mask_frame = masks[test_frame] > 0 
    mask_frame = skmorph.binary_closing(mask_frame, skmorph.disk(1))
    mask_frame = binary_fill_holes(mask_frame)

    contour = skmeasure.find_contours(mask_frame,0)

    contour = contour[np.argmax([len(cc) for cc in contour])]
    contour = contour[:-1] # need to do this ----> don't include the last point. 

    mask_dist_tform = unwrap3D_segmentation.sdf_distance_transform(mask_frame)
    mask_dist_tform_gradients = np.array(np.gradient(mask_dist_tform)).transpose(1,2,0) # 2D gradients. (non-normalized, therefore taking into account curvature)


    """ 
    generate reference contour and calculate d 

    """

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
    print(contour_evolve[0])
    print(contour_evolve.shape)
    plt.figure(figsize=(10,10))
    plt.imshow(raw[test_frame], cmap='gray')
    plt.plot(contour_evolve[...,0][:,1], contour_evolve[...,0][:,0],'r',lw=1)
    plt.plot(contour_evolve[...,1][:,1], contour_evolve[...,1][:,0],'g',lw=1)
    plt.plot(contour_evolve[...,2][:,1], contour_evolve[...,2][:,0],'y',lw=1)
    # for ii in np.arange(contour_evolve.shape[-1])[:]:
    #     plt.plot(contour_evolve[...,ii][:,1], 
    #                 contour_evolve[...,ii][:,0], lw=3)
    #     break
    #plt.plot(contour_evolve[:,1], contour_evolve[:,0], 'g', lw=1)
    plt.savefig(os.path.join(savefolder,'reference_contour.png'))
    plt.show()

    reference_n = 2
    d, ac_d = calculate_d(contour_evolve, reference_n)
    

    """
    plot the first 4 contours for sanity check

    """
    plot_d(raw[test_frame], d, contour_evolve, [i for i in range(5)], savefolder=savefolder, show_img=True)


    


    """
    2024.8.22
    New for mapping the contour to the vertices

    """
    print("contour_evolve shape", contour_evolve.shape)
    print("contour", contour_evolve[0,:,0])
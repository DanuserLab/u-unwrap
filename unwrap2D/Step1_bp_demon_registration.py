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
            --> currently using kdTree; Can be improved by cMCF***
            -- If there's redundance in one frame --> perform linear interpolation between points
        - Reorder propagated_boundary (using fixed_boundary) to make sure there's no intersection

Step 3:
    Interpolation on the UV coordinate
        - Map the first frame boundary points to the UV coordinate
        - Use propagated_boundary to interpolate the UV coordinate of other frames
    


This file contains the Step 1

"""
import SimpleITK as sitk
import numpy as np
from tifffile import imread, imwrite
import os
import sys
from tqdm import tqdm
from copy import deepcopy


"""
Felix's unwrap3D code


"""
def rescale_img(img, p12, eps=1e-12):
    img_ = imadjust(img, p12[0],p12[1])
    img_ = (img_ - img_.min()) / (img_.max() - img_.min() + eps)
    return img_


def imadjust(vol, p1, p2): 
    from skimage.exposure import rescale_intensity
    # this is based on contrast stretching and is used by many of the biological image processing algorithms.
    p1_, p2_ = np.percentile(vol, (p1,p2))
    vol_rescale = rescale_intensity(vol, in_range=(p1_,p2_))
    return vol_rescale

def smooth_and_resample(image, shrink_factors, smoothing_sigmas):
    """ Utility function used in :func:`unwrap3D.Registration.registration.multiscale_demons` for generating multiscale image pyramids for registration based on the SimpleITK library

    see SimpleITK notebooks, https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/66_Registration_Demons.html

    Parameters
    ----------
    image : SimpleITK image
        The image we want to resample.
    shrink_factors : scalar or array
        Number(s) greater than one, such that the new image's size is original_size/shrink_factor.
    smoothing_sigma(s): scalar or array
        Sigma(s) for Gaussian smoothing, this is in physical units, not pixels.

    Returns
    -------
    image_resample : SimpleITK image
        Image which is a result of smoothing the input and then resampling it using the given sigma(s) and shrink factor(s).
    """
    import SimpleITK as sitk
    import numpy as np 

    if np.isscalar(shrink_factors):
        shrink_factors = [shrink_factors]*image.GetDimension()
    if np.isscalar(smoothing_sigmas):
        smoothing_sigmas = [smoothing_sigmas]*image.GetDimension()

    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigmas)
    
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz/float(sf) + 0.5) for sf,sz in zip(shrink_factors,original_size)]
    new_spacing = [((original_sz-1)*original_spc)/(new_sz-1) 
                   for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]

    image_resample = sitk.Resample(smoothed_image, new_size, sitk.Transform(), 
                         sitk.sitkLinear, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0.0, 
                         image.GetPixelID())
    
    return image_resample


def multiscale_demons(registration_algorithm,
                      fixed_image, moving_image, initial_transform = None, 
                      shrink_factors=None, smoothing_sigmas=None):
   
    import SimpleITK as sitk
    import numpy as np 

    # Create image pyramid in a memory efficient manner using a generator function.
    # The whole pyramid never exists in memory, each level is created when iterating over
    # the generator.
    def image_pair_generator(fixed_image, moving_image, shrink_factors, smoothing_sigmas):
        end_level = 0
        start_level = 0
        if shrink_factors is not None:
            end_level = len(shrink_factors)
        for level in range(start_level, end_level):
            f_image = smooth_and_resample(fixed_image, shrink_factors[level], smoothing_sigmas[level])
            m_image = smooth_and_resample(moving_image, shrink_factors[level], smoothing_sigmas[level])
            yield(f_image, m_image)
        yield(fixed_image, moving_image)
    
    # Create initial displacement field at lowest resolution. 
    # Currently, the pixel type is required to be sitkVectorFloat64 because 
    # of a constraint imposed by the Demons filters.
    if shrink_factors is not None:
        original_size = fixed_image.GetSize()
        original_spacing = fixed_image.GetSpacing()
        s_factors =  [shrink_factors[0]]*len(original_size) if np.isscalar(shrink_factors[0]) else shrink_factors[0]
        df_size = [int(sz/float(sf) + 0.5) for sf,sz in zip(s_factors,original_size)]
        df_spacing = [((original_sz-1)*original_spc)/(new_sz-1) 
                      for original_sz, original_spc, new_sz in zip(original_size, original_spacing, df_size)]
    else:
        df_size = fixed_image.GetSize()
        df_spacing = fixed_image.GetSpacing()
 
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(initial_transform, 
                                                                       sitk.sitkVectorFloat64,
                                                                       df_size,
                                                                       fixed_image.GetOrigin(),
                                                                       df_spacing,
                                                                       fixed_image.GetDirection())
    else:
        initial_displacement_field = sitk.Image(df_size, sitk.sitkVectorFloat64, fixed_image.GetDimension())
        initial_displacement_field.SetSpacing(df_spacing)
        initial_displacement_field.SetOrigin(fixed_image.GetOrigin())
 
    # Run the registration.            
    # Start at the top of the pyramid and work our way down.    
    for f_image, m_image in image_pair_generator(fixed_image, moving_image, shrink_factors, smoothing_sigmas):
        initial_displacement_field = sitk.Resample(initial_displacement_field, f_image)
        initial_displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)
    return sitk.DisplacementFieldTransform(initial_displacement_field)


def SITK_Symmetric_demons_registration(img1, img2,
                                       imtype=16,
                                       p12=(2,99.8),
                                       rescale_intensity=True,  
                                       centre_tfm_model='geometry', 
                                       n_iters = 25, 
                                       smooth_displacement_field = True,
                                       smooth_alpha=.8,
                                       shrink_factors = [2.,1.], 
                                       smoothing_sigmas = [1.,1.],
                                       eps=1e-12): 
                                        
  
    im1_ = img1.copy()
    im2_ = img2.copy()
  
    img1 = sitk.GetImageFromArray(im1_, isVector=False)
    img2 = sitk.GetImageFromArray(im2_, isVector=False)
        
    # a) initial transform 
    # translation.
    if centre_tfm_model=='geometry':
        translation_mode = sitk.CenteredTransformInitializerFilter.GEOMETRY
    if centre_tfm_model=='moments':
        translation_mode = sitk.CenteredTransformInitializerFilter.MOMENTS
    initial_transform = sitk.CenteredTransformInitializer(img1, 
                                                          img2, 
                                                          sitk.Euler2DTransform(),
                                                          translation_mode)

    # a) demons transform (best to have corrected out any rigid transforms a priori) 
     # Select a Demons filter and configure it.
    
    demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
 
    # set the number of iterations
    demons_filter.SetNumberOfIterations(n_iters) # 5 for less. # long time for 20? 
    # Regularization (update field - viscous, total field - elastic).
    demons_filter.SetSmoothDisplacementField(smooth_displacement_field)
    demons_filter.SetStandardDeviations(smooth_alpha)
    
    # run the registration and return the final transform parameters
    final_tfm = multiscale_demons(registration_algorithm=demons_filter, 
                                  fixed_image = img1,
                                  moving_image = img2,
                                  initial_transform = initial_transform,
                                  shrink_factors = shrink_factors, # did have 2 here. -> test, can we separate the  # do at the same scale. 
                                  smoothing_sigmas = smoothing_sigmas) # set smoothing very low, since we want it to zone in on interesting features. 
    # check again how this is parsed .

    return final_tfm


def transform_img_sitk(vol, tfm):
    r""" One-stop function for applying any SimpleITK transform to an input image. Linear interpolation is used.  
    
    Parameters
    ----------
    vol: array
        input image as a numpy array     
    tfm: SimpleITK.Transform
        A simpleITK transform instance such as that resulting from using the simpleITK registration functions in this module.
    
    Returns
    -------
    v_transformed : array
        resulting image after applying the given transform to the input, returned as a numpy array 

    """
    import SimpleITK as sitk

    v = sitk.GetImageFromArray(vol, isVector=False)
    v_transformed = sitk.Resample(v, 
                                  v, 
                                  tfm, # this should work with all types of transforms.
                                  sitk.sitkLinear, 
                                  0.0, 
                                  v.GetPixelID())
    v_transformed = sitk.GetArrayFromImage(v_transformed) # back to numpy format. 
    
    return v_transformed




def extract_displacement_field(transform, reference_image):
    """
    Convert a SimpleITK displacement field transform into a displacement vector field.

    Parameters:
        transform (sitk.Transform): The displacement field transform from registration.
        reference_image (sitk.Image): The fixed image used in registration.

    Returns:
        displacement_field (np.array): A NumPy array (H, W, 2) containing displacement vectors.
    """
    displacement_field = sitk.TransformToDisplacementField(transform, 
                                                           sitk.sitkVectorFloat64, 
                                                           reference_image.GetSize(), 
                                                           reference_image.GetOrigin(), 
                                                           reference_image.GetSpacing(), 
                                                           reference_image.GetDirection())
    
    #displacement_array = sitk.GetArrayFromImage(displacement_field)
    
    return displacement_field  # Shape: (Height, Width, 2) containing (dx, dy)





"""
Master Function for performing frame by frame registration
Return registered images, original images and displacement field
Also calculate the MSE between fixed images and registerd images
"""



def frame_by_frame_demon_registration(img_stack, 
                                      p12 = (0,99.9),
                                      shrink_factor = [2, 1.],
                                      smooth_alpha = 2,
                                      smoothing_sigmas=[1,1],
                                      n_iters = 25,
                                      imtype=16.,
                                      eps = 1e-12):




    ori_img_stack = []
    registered_stack = []
    displacement_fields_stack = []
    mses = []

    for ii in tqdm(range(len(img_stack) - 1)):
        
        im1_ = deepcopy(img_stack[ii])
        im1_ = rescale_img(im1_, p12)
        ori_img_stack.append(im1_)
        
        if ii == 0:
            registered_stack.append(im1_)


        # rescale moving image
        im2_ = deepcopy(img_stack[ii + 1])   
        im2_ = rescale_img(im2_, p12)
        
    

        tfm = SITK_Symmetric_demons_registration(im1_, im2_,smooth_displacement_field = True, shrink_factors = shrink_factor,  \
                                                smooth_alpha = smooth_alpha, \
                                                n_iters=n_iters,
                                                smoothing_sigmas=smoothing_sigmas)

        im2_registered = transform_img_sitk(im2_, tfm)
        
        mse = np.mean((im2_registered - im1_) ** 2)
        
        fx = sitk.GetImageFromArray(im1_, isVector=False)
      

        displacement_field = extract_displacement_field(tfm, sitk.GetImageFromArray(im1_))
    

        displacement_fields_stack.append(sitk.GetArrayFromImage(displacement_field))
      
        mses.append(mse)
        
       
        registered_stack.append(im2_registered)
        
    ori_img_stack.append(im2_)    
    ori_img_stack = np.array(ori_img_stack).astype(np.float32)
    registered_stack = np.array(registered_stack).astype(np.float32)
    print("mean mses:", np.mean(mses))
    print("mse std:", np.std(mses))
    return registered_stack, ori_img_stack, displacement_fields_stack, np.mean(mses)




"""
Visualization Code for sanity check

"""

def save_demon_registration_overlay_plot(original_img, registered_img, save_dir, frame_idx=3):
    """
    Create an overlay plot of original and registered images and save it as PNG.
    
    Parameters:
        original_img (2D numpy array): Original grayscale image.
        registered_img (2D numpy array): Registered grayscale image.
        save_dir (str): Directory to save the figure.
        frame_idx (int): Index of the frame, used in the filename.
    """

  
    filename = f"frame_{frame_idx:03d}.png" 
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(15,15))
    plt.imshow(registered_img, cmap='Greens')
    plt.imshow(original_img, cmap='Reds', alpha=0.5)
 

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
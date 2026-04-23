import SimpleITK as sitk

from tifffile import imwrite

import os
import numpy as np

import matplotlib.pyplot as plt

import unwrap3D.Parameters.params as parameters


## rotation
def multires_registration(fixed_image, moving_image, initial_transform, center,  threshold = 0):
    mask_fixed = sitk.BinaryThreshold(fixed_image, lowerThreshold=1e-8, upperThreshold=1e10, \
                                insideValue=1, outsideValue=0)
    mask_moving = sitk.BinaryThreshold(moving_image, lowerThreshold=1e-8, upperThreshold=1e10, \
                                insideValue=1, outsideValue=0)                            
 
    registration_method = sitk.ImageRegistrationMethod()
    #registration_method.SetMetricFixedMask(mask_fixed)
    #registration_method.SetMetricMovingMask(mask_moving)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50) 
    # registration_method.SetMetricAsMeanSquares()

    #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    #registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)

    registration_method.SetMetricSamplingPercentage(0.1)
    # registration_method.SetMetricSamplingPercentage(0.2)
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # registration_method.SetOptimizerAsExhaustive(numberOfSteps=[10, 0, 0],  stepLength= np.pi/720) 
    # registration_method.SetOptimizerAsExhaustive(numberOfSteps=[20, 20, 0],  stepLength= np.pi/720) 
    print("gradient")
    registration_method.SetOptimizerAsRegularStepGradientDescent(
    learningRate=1,
    minStep=1e-12,
    numberOfIterations=3000,
    gradientMagnitudeTolerance=1e-12
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
 

    final_transform = registration_method.Execute(fixed_image, moving_image)

    print(f"Final metric value: {registration_method.GetMetricValue()}")
    print(
        f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
    )
    return final_transform, registration_method.GetMetricValue()

def rotation_registration(fixed_img_np, moving_img_np, extra_imgs = None):
    # initialize the transform
    fixed_img = sitk.GetImageFromArray(fixed_img_np)
    moving_img = sitk.GetImageFromArray(moving_img_np)

    initial_transform = sitk.CenteredTransformInitializer(
    fixed_img,
    moving_img,
    sitk.Euler2DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # 
    size = fixed_img.GetSize()
    center_index = [s / 2.0 for s in size]
    center_physical = fixed_img.TransformContinuousIndexToPhysicalPoint(center_index)
    initial_transform.SetCenter(center_physical)
    #initial_transform.SetTranslation([0.0, 0.0])
    final_transform, metrics = multires_registration(fixed_img, moving_img, initial_transform, center_physical)

    resampled = sitk.Resample(
        moving_img,
        fixed_img,
        final_transform,
        sitk.sitkNearestNeighbor,
        #sitk.sitkBSpline, 
        0.0,
        moving_img.GetPixelID()
    )

    if extra_imgs is None:
        return sitk.GetArrayFromImage(resampled), metrics
    else:
        resampled_extra_imgs = []
        for imgs in extra_imgs:
            new_moving = sitk.GetImageFromArray(imgs)
            extra_resampled = sitk.Resample(
                new_moving,
                fixed_img,
                final_transform,
                sitk.sitkNearestNeighbor,
                #sitk.sitkBSpline, 
                0.0,
                new_moving.GetPixelID()
            )
            resampled_extra_imgs.append(sitk.GetArrayFromImage(extra_resampled))
        return sitk.GetArrayFromImage(resampled), metrics, resampled_extra_imgs



## translation using cmcf find center
from skimage import measure
import os
import sys
sys.path.append('/work/bioinformatics/s440708/unwrap2D/Unwrap2D/unwrap2D/')
import unwrap2D as unwrap2D_fns

def find_center(image, mask=None):
    import unwrap3D.Segmentation.segmentation as unwrap3D_segmentation
    from skimage import measure, morphology


        

    
    mask_dist_tform = unwrap3D_segmentation.sdf_distance_transform(mask)
    mask_dist_tform_gradients = np.array(np.gradient(mask_dist_tform)).transpose(1,2,0)

    contours = measure.find_contours(mask, level=0.5)
    contour = max(contours, key = len)
    contour_evolve = unwrap2D_fns.parametric_line_flow_2D(contour,
                                                external_img_gradient=mask_dist_tform_gradients, 
                                                E=None, 
                                                close_contour=True, 
                                                fixed_boundary = False, 
                                                lambda_flow=850, # adjusts the balance.  (decrease this to be more similar to original cell shape ), increase to be more like version 1
                                                step_size= 1, # adjusts the spacing between curves. 
                                                niters=50, 
                                                conformalize=False,
                                                eps=1e-12)
    last_contour = contour_evolve[...,-1]
    ymin, xmin = np.min(last_contour, axis = 0)
    ymax, xmax = np.max(last_contour, axis = 0)
    center = ((ymin+ymax)/2, (xmin+xmax)/2)
    return center


def center_translation(img, center):
    from scipy.ndimage import shift
    H, W = img.shape


    center_y, center_x = center
    target_y, target_x = H // 2, W // 2
    dy, dx = target_y - center_y, target_x - center_x

    centered_image = shift(img, shift=(dy, dx), order=0, prefilter=False, mode='nearest')
    center = (center[0] + dy, center[1] + dx)

    return centered_image, center, (dy, dx)

def img_translation(img, mask=None, cMCF_center = True):
    from skimage import measure, morphology


    def get_centroid_meanxy(mask):
        ys, xs = np.where(mask)  
        assert len(xs) != 0
        cx = xs.mean()
        cy = ys.mean()
        return (cy, cx)

    if mask is None:
        mask = img > 0
        labels = measure.label(mask)
        assert( labels.max() != 0 ) 
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        mask = morphology.remove_small_holes(largestCC, area_threshold=50)

    if cMCF_center:
        center = find_center(img, mask=mask)
    else:
        center = get_centroid_meanxy(mask)

    centered_image, center, (dy, dx) = center_translation(img, center)
    # plt.imshow(centered_image, cmap='jet')
    # plt.scatter(center[1], center[0], color='red')
    # plt.show()
    return centered_image, center, (dy, dx)
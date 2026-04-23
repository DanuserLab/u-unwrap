import numpy as np
import matplotlib.pyplot as plt
import os

def generate_binary_mask(img_size = 1280, radius = 511):

    disk_mask = np.zeros((img_size, img_size), dtype=np.uint8)

    center_x, center_y = img_size // 2, img_size // 2

    y, x = np.ogrid[:img_size, :img_size]

    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2


    disk_mask[mask] = 1


    return disk_mask

def plot_intensity_origin(segment_masks, 
                          unwrap_params_stack,
                          unwrap_imgs_stack,
                          unwrap_valid_mask_stack,
                          save_path = None, 
                          cmap=None,
                          vmin=None,
                          vmax=None,
                          interpolation=None,
                          radius = 127, 
                          layer=1, 
                          theta_num=60, 
                          colorbar_fraction=0.03,  # Default is 0.046
                          colorbar_pad=0.04, 
                            ):
    """
    Simple plot with adjustable colorbar size using fraction and pad.

    Parameters:
    - all_intensity (list of lists or arrays): Intensity data.
    - layer (int): The layer to plot.
    - theta_num (int): Number of windows per layer.
    - colorbar_fraction (float): Fraction of the original axes to use for the colorbar.
    - colorbar_pad (float): Padding between the main plot and the colorbar.
    - cmap (str): Colormap to use for the intensity plot.

    Returns:
    - intensity (2D NumPy array): Computed intensity values.
    """
    # Compute intensity matrix
    from unwrap3D.Unzipping.unzip import surface_area_uv
    import skimage.morphology as skmorph
    n_frame = len(unwrap_params_stack)
    intensity = np.zeros((theta_num, n_frame))
    
    for ii, (unwrap_params, unwrap_img) in enumerate(zip(unwrap_params_stack, unwrap_imgs_stack)):
        unwrap_valid_mask =  unwrap_valid_mask_stack[ii]
        u,v = unwrap_params.shape[:2]
        zeros = np.zeros((u, v, 1), dtype=unwrap_params.dtype)

        unwrap_params_safe = unwrap_params.copy()
        unwrap_params_safe[~np.isfinite(unwrap_params_safe)] = 0
        unwrap_params_safe[np.isnan(unwrap_params_safe)] = 0

        unwrap_params_3d = np.concatenate((zeros, unwrap_params_safe), axis=2)
        dS_dudv, total_dS_dudv = surface_area_uv(unwrap_params_3d, eps=1e-12, pad=False)
        dS_dudv[np.isinf(dS_dudv)] = 0
        disk_mask = generate_binary_mask(unwrap_img.shape[0], radius = radius)
        dS_dudv[np.logical_not(skmorph.binary_erosion(unwrap_valid_mask, skmorph.disk(1)))] = 0
        dS_dudv = disk_mask * dS_dudv
        w_i = dS_dudv / total_dS_dudv
        lower_percentile = np.percentile(w_i, 0)
        upper_percentile = np.percentile(w_i, 99.9)

        w_i = np.clip(w_i, lower_percentile, upper_percentile)
        corrected_unwrap_img = unwrap_img * w_i
    
        # masks = segment_masks
        s = theta_num * (layer - 1)
        e = theta_num * layer
        # idx_list = [ii for ii in range(s,e)]
        for idx in range(s,e):
        
            mask = (segment_masks == (idx+1)).astype(np.int16)
    
            jj = idx % theta_num
            
            numerator = np.nansum(corrected_unwrap_img[mask>0]) ### frame
            denominator = np.nansum(w_i[mask>0])

            
            # window = window[window>0]
            #assert np.all(samples > 0)
            intensity[jj, ii] =  numerator/(denominator+1e-20)
        
    # Create main plot

    fig, ax = plt.subplots(figsize=(10, 8))
    imshow_kwargs = dict(cmap=cmap)

    if cmap is not None:
        imshow_kwargs["cmap"] = cmap
    if vmin is not None:
        imshow_kwargs["vmin"] = vmin
    if vmax is not None:
        imshow_kwargs["vmax"] = vmax
    if interpolation is not None:
        imshow_kwargs["interpolation"] = interpolation

    cax = ax.imshow(intensity, **imshow_kwargs)


    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Window', fontsize=12)
    ax.set_title(f'Corrected Mean Intensity of Layer {layer} Each Window in Each Frame', fontsize=14)
    
    # Create colorbar with adjusted size
    cb = plt.colorbar(cax, ax=ax, fraction=colorbar_fraction, pad=colorbar_pad)
    cb.set_label('Mean Intensity', fontsize=12)
    cb.ax.tick_params(labelsize=10)
    if save_path:
        plt.savefig(os.path.join(save_path, f'layer_{layer}.svg'))
    plt.show()
    plt.close()
    return intensity


def plot_intensity_ratio(all_intensity, save_path = None, layer=1, theta_num=59, 
                          colorbar_fraction=0.03,  # Default is 0.046
                          colorbar_pad=0.04, 
                          cmap='rainbow'):
    """
    Simple plot with adjustable colorbar size using fraction and pad.

    Parameters:
    - all_intensity (list of lists or arrays): Intensity data.
    - layer (int): The layer to plot.
    - theta_num (int): Number of windows per layer.
    - colorbar_fraction (float): Fraction of the original axes to use for the colorbar.
    - colorbar_pad (float): Padding between the main plot and the colorbar.
    - cmap (str): Colormap to use for the intensity plot.

    Returns:
    - intensity (2D NumPy array): Computed intensity values.
    """
    # Compute intensity matrix
    intensity = np.zeros((theta_num, len(all_intensity)))  # N of windows, N of frames
    for i, frame in enumerate(all_intensity):
        s = theta_num * (layer - 1)
        e = theta_num * layer
        for j, window in enumerate(frame[s:e]):
            window = window[window>0]
            assert np.all(window > 0)
            intensity[j, i] = np.sum(window)
    
    # Create main plot
    # fig, ax = plt.subplots(figsize=(10, 8))
    # cax = ax.matshow(intensity, cmap=cmap)
    # ax.set_xlabel('Frame', fontsize=12)
    # ax.set_ylabel('Window', fontsize=12)
    # ax.set_title(f'Corrected Mean Intensity of Layer {layer} Each Window in Each Frame', fontsize=14)
    
    # # Create colorbar with adjusted size
    # cb = plt.colorbar(cax, ax=ax, fraction=colorbar_fraction, pad=colorbar_pad)
    # cb.set_label('Mean Intensity', fontsize=12)
    # cb.ax.tick_params(labelsize=10)
    # if save_path:
    #     plt.savefig(os.path.join(save_path, f'layer_{layer}'))
    # plt.show()
    return intensity

    
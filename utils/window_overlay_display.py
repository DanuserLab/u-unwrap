import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy.ma as ma
from unwrap2D import uv_to_yx_pts2D

def window_display(window, THETA_NUM, R_NUM, output_folder, frame_n, file_name = None, color_track = True,img = None, img_gray = None, color_layer = True,legend=True):
    import cv2
    import matplotlib.colors as mcolors
    from skimage.measure import find_contours
    from skimage.draw import polygon
    import skimage.morphology as skmorph
    fig, ax = plt.subplots(figsize=(6, 6))

    fig.patch.set_facecolor('white')  
    ax.set_facecolor('white')
    if img is not None:
        if img.shape[-1] == 3:
            img_copy = np.zeros_like(img[:,:,0])
            img = img.astype(np.float32)
            if img.max() > 1:
                img = img / 255.0
            img[img_gray==0] = [np.nan,np.nan,np.nan]
        else:
            img_copy = np.zeros_like(img)
   
        ax.imshow(img, cmap='magma')
    else:
        img_copy = np.zeros((1024,1024))


    colors = sns.color_palette("colorblind", n_colors=5)
    k = max(1, THETA_NUM // 3)
    theta_marks = {0, k, min(2 * k, THETA_NUM - 1)}
    

    _, _, segments_masks, _ = window.split_window(r_num=R_NUM, theta_num=THETA_NUM)

    all_masks = [segments_masks == ii for ii in range(1, int(np.max(segments_masks)+1))]
    all_xy = []
    for i, mask in enumerate(all_masks):
        layer_index = i // THETA_NUM 
        
        theta_idx   = i % THETA_NUM  

        if color_layer:
            color = colors[layer_index % len(colors)]
        else:
            color = colors[layer_index % len(colors)]
        # mask = mask.astype(np.uint8)
        # kernel = np.ones((1,1), np.uint8)
        if not np.any(mask):
            continue
        mask = largest_component_vol(mask)
        mask_eroded = mask
        mask_eroded = skmorph.binary_erosion(mask, skmorph.disk(1))
        # radius = 1
        # ksize = (int(2 * radius + 1), int(2 * radius + 1))
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
        # mask_eroded = cv2.erode(mask, kernel, iterations=1)
        contours = find_contours(mask_eroded)

        if not contours:
            continue

        contour = max(contours, key=len)
       
        #uv = np.round(contour).astype(int)
        
        #xy = window.unwrap_params[uv[:, 0], uv[:, 1]]
        # mask_valid = ~np.all(xy == 0, axis=1)
        # xy = xy[mask_valid]
 
        xy = uv_to_yx_pts2D(contour, window.unwrap_params)
        all_xy.append(xy)
        rr, cc = polygon(xy[:, 0], xy[:, 1], shape=img_copy.shape[:2])
        img_copy[rr, cc] = i + 1
        
        if theta_idx == 0:
            # print(layer_index)
            if color_layer:
                
            #ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=0.5, label = f"Layer {int(idx/THETA_NUM)}")
                ax.plot(xy[:, 1], xy[:, 0], color=color, linewidth=0.5,label = f"Layer {layer_index + 1}")
                
                    
            else:

                ax.plot(xy[:, 1], xy[:, 0], color=color, linewidth=0.5)
        else:
            #ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=0.5)
            ax.plot(xy[:, 1], xy[:, 0], color=color, linewidth=0.5)
       
        if color_track and theta_idx % 5 == 0:
                    
            ax.fill(xy[:, 1], xy[:, 0], color="gray", alpha=0.5, zorder=0) 
        # ax.plot(xy[:, 1], xy[:, 0], color=color, linewidth=0.5, alpha=1)
        if layer_index == 0 and theta_idx in theta_marks:
            y_c, x_c = np.nanmean(xy, axis=0)           
            ax.text(x_c, y_c, f"{theta_idx + 1}", color='black', fontsize=15, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.5))

     
    ax.set_axis_off()
    if legend:
        ax.legend()
    if not img:
        ax.invert_yaxis()
    os.makedirs(output_folder, exist_ok=True)
    
    if file_name is not None:
        svg_filename = f'{file_name}_frame{frame_n:03d}.svg'
    else:
        svg_filename = f'frame_{frame_n:03d}.svg'  
    svg_path = os.path.join(output_folder, svg_filename)
    
    plt.show()
    fig.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return img_copy, all_xy


def largest_component_vol(vol_binary, connectivity = 1):
    from skimage.measure import label, regionprops
    import numpy as np
    vol_binary_labelled = label(vol_binary, connectivity=connectivity)
    vol_binary_props = regionprops(vol_binary_labelled)
    vol_binary_vols = [re.area for re in vol_binary_props]
    vol_binary = vol_binary_labelled == (np.unique(vol_binary_labelled)[1:][np.argmax(vol_binary_vols)])
    return vol_binary

def draw_label_contours(img, img_gray, img_label, THETA_NUM, savefolder, name):
    import cv2
    labels = np.unique(img_label)
    labels = labels[labels > 0]
    labels = np.sort(labels)
    n_labels = len(labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    # fig.patch.set_facecolor('white')  
    # ax.set_facecolor('white')
    # img = img.astype(np.float32)
    # if img.max() > 1:
    #     img = img / 255.0
    # img[img_gray==0] = [np.nan,np.nan,np.nan]


    ax.imshow(img)
    colors = sns.color_palette("colorblind", n_colors=3)

    for idx, lbl in enumerate(labels):
        mask = (img_label == lbl).astype(np.uint8)
        radius = 1
        ksize = (int(2 * radius + 1), int(2 * radius + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
        #mask_eroded = cv2.erode(mask, kernel, iterations=1)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
        contour = max(contours,key=len)
        if not contours:
            continue

        if idx < THETA_NUM:
            color = colors[0]
        elif idx >= n_labels - THETA_NUM:
            color = colors[2]
        else:
            color = colors[1]
        if idx % THETA_NUM == 0:
     
            #ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=0.5, label = f"Layer {int(idx/THETA_NUM)}")
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], color=color, linewidth=0.5,label = f"Layer {int(idx/THETA_NUM) + 1}")
        else:
            #ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=0.5)
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], color=color, linewidth=0.5)

    ax.set_axis_off()
    ax.legend(loc='upper left')
    if img is None:
        ax.invert_yaxis() 
    plt.tight_layout()
    plt.savefig(os.path.join(savefolder, name))
    plt.show()

def calculate_uv_area_portion(unwrap_params, unwrap_valid_mask):
    import skimage.morphology as skmorph
    from unwrap3D.Unzipping.unzip import surface_area_uv
    u,v = unwrap_params.shape[:2]
    zeros = np.zeros((u, v, 1), dtype=unwrap_params.dtype)

    unwrap_params_3d = np.concatenate((zeros, unwrap_params), axis=2)
    dS_dudv, total_dS_dudv = surface_area_uv(unwrap_params_3d, eps=1e-12, pad=False)
    dS_dudv[np.isinf(dS_dudv)] = 0
    # dS_dudv[np.logical_not(unwrap_valid_mask)]=0

    dS_dudv[np.logical_not(skmorph.binary_erosion(unwrap_valid_mask, skmorph.disk(1)))] = 0
    total_dS_dudv = np.sum(dS_dudv)

    w = dS_dudv/total_dS_dudv

    low = np.percentile(w, 0)
    high = np.percentile(w, 99.5)
    w_ = np.clip(w, low, high)
    # plt.imshow(w_, cmap='coolwarm')
    # plt.title("area proportion (dA/A)")
    # plt.colorbar()

    return w_

def plot_window_area_portion(all_masks, 
                          unwrap_params_stack,
                          unwrap_imgs_stack,
                          unwrap_valid_mask_stack,
                          img_size = (256,256),
                          save_path = None, vmin = 0, vmax=1, radius = 127, theta_num=59, 
                          colorbar_fraction=0.03, 
                          colorbar_pad=0.04, 
                          cmap='rainbow'):
    
    # Compute intensity matrix
    from unwrap3D.Unzipping.unzip import surface_area_uv
    import skimage.morphology as skmorph
    from skimage.measure import find_contours
    from unwrap2D import uv_to_yx_pts2D
    from skimage.draw import polygon
    import skimage.morphology as skmorph
    n_frame = len(unwrap_params_stack)
    intensity = np.zeros((theta_num, n_frame))
    
    window_uv_area_portion_stack = []
    window_xy_area_portion_stack = []
    window_xy_area_portion_img_stack = []
    window_ratio_disk_stack = []
    ratio_value_stack = []
    for ii, (unwrap_params, unwrap_img) in enumerate(zip(unwrap_params_stack, unwrap_imgs_stack)):
 
        unwrap_valid_mask =  unwrap_valid_mask_stack[ii]
        
        u,v = unwrap_params.shape[:2]
        

        window_area_portion = np.zeros((u, v), dtype=unwrap_params.dtype)

        zeros = np.zeros((u, v, 1), dtype=unwrap_params.dtype)

        unwrap_params_3d = np.concatenate((zeros, unwrap_params), axis=2)
        dS_dudv, total_dS_dudv = surface_area_uv(unwrap_params_3d, eps=1e-12, pad=False)
        dS_dudv[np.isinf(dS_dudv)] = 0
        disk_mask = generate_binary_mask(unwrap_img.shape[0], radius = radius)
        eroded_valid_mask = skmorph.binary_erosion(unwrap_valid_mask, skmorph.disk(1))
        dS_dudv[np.logical_not(eroded_valid_mask)] = 0
        dS_dudv = disk_mask * dS_dudv
        total_dS_dudv = np.sum(dS_dudv)
        w_i = dS_dudv / total_dS_dudv
        lower_percentile = np.percentile(w_i, 0)
        upper_percentile = np.percentile(w_i, 99.5)

        w_i = np.clip(w_i, lower_percentile, upper_percentile)
        # corrected_unwrap_img = unwrap_img * w_i
    
        masks = all_masks[ii]

        
        # s = theta_num * (layer - 1)
        # e = theta_num * layer
        SANITY_CHECK = 0  
        all_xy_length = []  
        window_uv_area_portion = []
        window_xy_area_portion_img = np.zeros(img_size)
      
        for jj, mask in enumerate(masks):
            
            area_sum = np.nansum(w_i[mask>0]) 
            window_uv_area_portion.append(area_sum)
            window_area_portion[mask>0] = area_sum

            if jj < theta_num:
                mask = mask & eroded_valid_mask
            #     mask = skmorph.binary_erosion(mask, skmorph.disk(1))
            contours = find_contours(mask)
            if not contours:
                continue
            contour = max(contours, key=len)
            xy = uv_to_yx_pts2D(contour, unwrap_params)
            

            rr, cc = polygon(xy[:, 0], xy[:, 1], shape=img_size)
            window_xy_area_portion_img[rr, cc] = len(rr)
            # plt.imshow(window_xy_area_portion_img)
            # plt.show()
            all_xy_length.append(len(rr))

            # window = window[window>0]
            #assert np.all(samples > 0)
            # intensity[jj, ii] =  numerator/(denominator+1e-20)
        
        # calculate xy area portion normalized by the overall area of cell
      
        window_xy_area_portion = all_xy_length/np.sum(all_xy_length)
        window_xy_area_portion_img = window_xy_area_portion_img/np.sum(all_xy_length)
        ratio_disk_img = np.zeros_like(window_area_portion)
        ratio_values = []
        for jj, mask in enumerate(masks):
            if jj < theta_num:
                mask = mask & eroded_valid_mask
            xy_portion = window_xy_area_portion[jj]
            if xy_portion == 0:
                xy_portion = 1e-6
            # disk divided by xy
            ratio_disk_img[mask>0] = xy_portion/ (window_area_portion[mask>0] + 1e-12)
            if jj >= theta_num:
                ratio_values.append(ratio_disk_img[mask>0][0])
        
        window_xy_area_portion_img_stack.append(window_xy_area_portion_img)
        window_xy_area_portion_stack.append(window_xy_area_portion)
        window_uv_area_portion_stack.append(window_area_portion)
        window_ratio_disk_stack.append(ratio_disk_img)
        ratio_value_stack.append(ratio_values)

    return window_uv_area_portion_stack, window_xy_area_portion_img_stack, window_xy_area_portion_stack, window_ratio_disk_stack, ratio_value_stack



def generate_binary_mask(img_size = 1280, radius = 511):

    disk_mask = np.zeros((img_size, img_size), dtype=np.uint8)

    center_x, center_y = img_size // 2, img_size // 2

    y, x = np.ogrid[:img_size, :img_size]

    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2


    disk_mask[mask] = 1


    return disk_mask





"""
uRegister window plot

"""
def load_cell_windows_layers(mat_path):
    from scipy.io import loadmat
    windows = loadmat(mat_path)
    windows_layers = windows['windows'][0]
    return windows_layers

def plot_windows_tab20(
    mat_path, THETA_NUM, R_NUM,
    img=None, output_folder='.', frame_n=0, save_name=None,
    show_layer_labels=True
):
    windows_layers = load_cell_windows_layers(mat_path)  



  
    def color_by_layer(layer_idx): 
        colors = sns.color_palette("colorblind", n_colors=5) 
        idx = layer_idx % 5
        return colors[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')



    window_num = len(windows_layers)
    k = max(1, window_num // 3)
    theta_marks = {0, k, min(2 * k, window_num - 1)}
    for i, layers in enumerate(windows_layers):
        theta_idx = i % window_num 
        layers = layers[0] 
        for layer_idx, layer_polys in enumerate(layers):
            lines = layer_polys[0]
            col = color_by_layer(layer_idx)
            all_x = []
            all_y = []
            for line in lines:
                line = np.asarray(line)
        

                if line.ndim == 2 and line.shape[0] == 2 and line.shape[1] > 1:
                    y = line[0].ravel()
                    x = line[1].ravel()
                    all_y.extend(y)
                    all_x.extend(x)
                    # m = ~(np.isnan(x) | np.isnan(y))
                    # if np.count_nonzero(m) > 1:
                    #     new_lines = np.column_stack([x[m], y[m]])
                    #     all_x.extend(x[m])
                    #     all_y.extend(y[m])
                    ax.plot(y, x, linewidth=0.6, color=col)
            
            if theta_idx % 5 == 0 and len(all_x) > 1 and len(all_y) > 1:
                ax.fill(all_y, all_x, color="gray", alpha=0.5, zorder=0)  
    
            if layer_idx == 0 and theta_idx in theta_marks:
                # all_xy = [x for sublist in layer_polys for x in sublist]
                y_c = np.nanmean(all_y)
                x_c = np.nanmean(all_x)
                #y_c, x_c = np.nanmean(all_lines, axis=0)           
                ax.text(y_c, x_c, f"{theta_idx + 1}", color='black', fontsize=25, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

    # if layer_handles:
    #     ax.legend(handles=[layer_handles[k] for k in sorted(layer_handles)], loc='best', frameon=False)

    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    ax.invert_yaxis()  
    os.makedirs(output_folder, exist_ok=True)
    if save_name is not None:
        svg_path = os.path.join(output_folder, f'{save_name}_frame_{frame_n:03d}.svg')
    else:
        svg_path = os.path.join(output_folder, f'frame_{frame_n:03d}.svg')
    fig.savefig(svg_path, format='svg', dpi=300, pad_inches=0)
    plt.show()
    return svg_path, windows_layers
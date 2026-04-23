import sys
import os
import pickle
sys.path.append("../unwrap2D/")
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from tqdm import tqdm
from Step0_rotation_translation_registration import img_translation, rotation_registration
from Step1_bp_demon_registration import frame_by_frame_demon_registration
from Step2_bp_forwardeuler_GVF import boundary_propagation_with_anomaly_detection, boundary_propagation_with_anomaly_detection_fix_boundary, \
                                    save_matplot_video, reorder_igl_boundary_stack, extract_igl_boundary_stack
from Step3_uv_interpolation import generate_bdy_uv, interpolate_2nn_stack
from Step4_run_unwrap2D import parallel_unwrap2D, parallel_resample_img
import unwrap2D as unwrap2D_fns

sys.path.append("../utils/")

from plot_functions import save_overlay_plot
from scipy.ndimage import shift
import argparse
from skimage.morphology import remove_small_holes, remove_small_objects
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.morphology import disk, binary_opening, binary_closing

def step0_rigid_registration_center(img_stack, mask_stack, *additional_stacks, vis_save_path):
    
    #step0_savepath = os.path.join(vis_save_path, "Step0_rigid_registration_centered")
    os.makedirs(vis_save_path, exist_ok=True)

    centered_img_stack = []
    centered_mask_stack = []
    centered_additional = [[] for _ in range(len(additional_stacks))]
    for ii, (img, mask) in enumerate(zip(img_stack, mask_stack)):
        centered_img, center, (dy, dx) = img_translation(img, cMCF_center=True)
        centered_mask = shift(mask, shift=(dy, dx), order=0, prefilter=False, mode='nearest')
        filename = os.path.join(vis_save_path,  f"{ii+1:03}.tiff")
        imwrite(filename, centered_img)
        centered_img_stack.append(centered_img)
        centered_mask_stack.append(centered_mask)

        for idx, stack in enumerate(additional_stacks):
            extra_frame = stack[ii]
            centered_extra = shift(extra_frame, shift=(dy, dx), order=0, prefilter=False, mode='nearest')
            centered_additional[idx].append(centered_extra)

    centered_img_stack = np.array(centered_img_stack)
    centered_mask_stack = np.array(centered_mask_stack)


    
    return (centered_img_stack, centered_mask_stack, *tuple(np.array(res) for res in centered_additional))



def step0_rotation_registration(img_stack, mask_stack, vis_save_path, rerun, *additional_stacks, switch_n=5, extra_names=None):
   
   
    if (not rerun) and os.path.exists(os.path.join(vis_save_path, "Step0_rigid_registration","rigid_registered_mask.tif")):
        rigid_registered_stack = imread(os.path.join(vis_save_path, "Step0_rigid_registration","rigid_registered_cell.tif"))
        rigid_registered_mask = imread(os.path.join(vis_save_path, "Step0_rigid_registration","rigid_registered_mask.tif"))
        rigid_registered_additional = [] 
        for name in extra_names:
            r = imread(os.path.join(vis_save_path, "Step0_rigid_registration", f"{name}.tif"))
            rigid_registered_additional.append(r)

        return (rigid_registered_stack, rigid_registered_mask, *tuple(np.array(r) for r in rigid_registered_additional)) 
    figure_savepath = os.path.join(vis_save_path, "Step0_rotation")
    os.makedirs(figure_savepath, exist_ok=True)
    

    rigid_registered_stack = [img_stack[0]]
    rigid_registered_mask = [mask_stack[0]]

    rigid_registered_additional = [[s[0]] for s in additional_stacks]
    

    fixed_img = rigid_registered_stack[-1].copy()

    for jj in np.arange(1, img_stack.shape[0]):
        
        if np.mod(jj, switch_n) == 0:
            fixed_img = rigid_registered_stack[-1].copy()
        
        moving_img = deepcopy(img_stack[jj])
        mask = mask_stack[jj]

        current_additional = [s[jj] for s in additional_stacks]
        
      
        resampled_np, metrics, registered_others = rotation_registration(
            fixed_img, moving_img, [mask] + current_additional
        )
        
      
        rigid_registered_stack.append(resampled_np)
        rigid_registered_mask.append(registered_others[0]) 

        for idx, reg_extra in enumerate(registered_others[1:]):
            rigid_registered_additional[idx].append(reg_extra)

     
        _plot_registration_result(fixed_img, moving_img, resampled_np, jj, figure_savepath)

  
    rigid_registered_stack = np.array(rigid_registered_stack)
    rigid_registered_mask = np.array(rigid_registered_mask)


    save_test_path = os.path.join(vis_save_path, "Step0_rigid_registration")
    os.makedirs(save_test_path, exist_ok=True)
    imwrite(os.path.join(save_test_path, "rigid_registered_cell.tif"), rigid_registered_stack)
    imwrite(os.path.join(save_test_path, "rigid_registered_mask.tif"), rigid_registered_mask)
    
    if extra_names:
        for name, stack in zip(extra_names, rigid_registered_additional):
            stack = np.array(stack)
            imwrite(os.path.join(save_test_path, f"{name}.tif"), stack)

  

    return (rigid_registered_stack, rigid_registered_mask, *tuple(np.array(r) for r in rigid_registered_additional))

def _plot_registration_result(fixed_img, moving_img, resampled_np, jj, save_path):
    plt.figure(figsize=(15,10))
    plt.subplot(121); plt.title('no-register')
    plt.imshow(fixed_img, cmap='Reds')
    plt.imshow(moving_img, cmap='Greens', alpha=0.5)
    plt.axis('off'); plt.grid('off')
    
    plt.subplot(122); plt.title('register')
    plt.imshow(fixed_img, cmap='Reds')
    plt.imshow(resampled_np, cmap='Greens', alpha=0.5)
    plt.axis('off'); plt.grid('off')
    
    plt.savefig(os.path.join(save_path, f"frame_{jj}.png"))
    plt.close()


def demon_results(ori_img_stack, vis_save_path, rerun):

    save_dir = os.path.join(vis_save_path,"test_results")
    
    if os.path.exists(os.path.join(save_dir, 'displacement_fields_stack.pkl')) and not rerun:
        with open(os.path.join(save_dir, 'displacement_fields_stack.pkl'), 'rb') as f:
            displacement_fields_stack = pickle.load(f)
        registered_stack = imread(os.path.join(save_dir, 'demon_registered.tif')).astype(np.float32)
    
        for ii, (ori, reg) in tqdm(enumerate(zip(ori_img_stack, registered_stack))):
            save_overlay_plot(ori_img_stack[ii-1], reg, save_dir, frame_idx=ii)
        print("displacement field loaded")
    else:
        registered_stack, ori_img_stack, displacement_fields_stack, mses = frame_by_frame_demon_registration(ori_img_stack, 
                                                                                                            shrink_factor = [8, 4, 2, 1.], 
                                                                                                            smooth_alpha = 1,
                                                                                                            smoothing_sigmas=[1, 1, 1, 1],
                                                                                                            n_iters=800)
        
        os.makedirs(save_dir, exist_ok=True)
        for ii, (ori, reg) in tqdm(enumerate(zip(ori_img_stack, registered_stack))):
            save_overlay_plot(ori_img_stack[ii-1], reg, save_dir, frame_idx=ii)

        with open(os.path.join(save_dir, 'displacement_fields_stack.pkl'), 'wb') as f:
            pickle.dump(displacement_fields_stack, f)

        imwrite(os.path.join(save_dir, 'demon_registered.tif'), registered_stack)
    
    return displacement_fields_stack


def run_GVF(displacement_fields_stack, rigid_registered_mask, registered_img_stack, vis_save_path, rerun):


    def _plot_boundary():
        save_name = f"forward_euler_after_rotation_registration_pure_resample_fix_disorder"
        save_folder = os.path.join(vis_save_path, "boundary_visualization")
        os.makedirs(save_folder, exist_ok=True)
        save_matplot_video(registered_img_stack, pure_resample, save_folder, save_name, colormap='gray',img_plot=True,scatter = False, plot_multiple_point = True)

        save_name = f"forward_euler_after_rotation_registration_100iter_lambda05_scatter_fix_disorder"
        save_folder = os.path.join(vis_save_path, "boundary_visualization")
        os.makedirs(save_folder, exist_ok=True)
        save_matplot_video(registered_img_stack, boundary_points_stack_euler, save_folder, save_name, colormap='gray',img_plot=True,scatter = False, plot_multiple_point = True)



    igl_unwarped_boundary_stack, mesh_2D_submesh_stack, _ = extract_igl_boundary_stack(rigid_registered_mask)

    output_dir = os.path.join(vis_save_path,"GVF")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(os.path.join(vis_save_path, 'boundary_points_stack_euler_fix_disorder.pkl')) and not rerun:
        with open(os.path.join(vis_save_path, 'boundary_points_stack_euler_fix_disorder.pkl'), 'rb') as f:
            boundary_points_stack_euler = pickle.load(f)
        
    
    else:
        
        boundary_points_stack_euler, pure_resample = boundary_propagation_with_anomaly_detection_fix_boundary(
                                                displacement_fields_stack, 
                                                rigid_registered_mask,
                                                igl_unwarped_boundary_stack, 
                                                output_dir,
                                                window_size = 6,      
                                                sigma_thresh = 3.0,  
                                                min_history = 5,       
                                                contour_distance_thres = 3.0,
                                                stats_plot = True)
    # save_name = f"forward_euler_after_rotation_registration_pure_resample_fix_disorder"
    # save_folder = os.path.join(vis_save_path, "boundary_visualization")
    # os.makedirs(save_folder, exist_ok=True)
    # save_matplot_video(registered_img_stack, pure_resample, save_folder, save_name, colormap='gray',img_plot=True,scatter = False, plot_multiple_point = True)

    # save_name = f"forward_euler_after_rotation_registration_100iter_lambda05_scatter_fix_disorder"
    # save_folder = os.path.join(vis_save_path, "boundary_visualization")
    # os.makedirs(save_folder, exist_ok=True)
    # save_matplot_video(registered_img_stack, boundary_points_stack_euler, save_folder, save_name, colormap='gray',img_plot=True,scatter = False, plot_multiple_point = True)

        with open(os.path.join(vis_save_path, 'boundary_points_stack_euler_fix_disorder.pkl'), 'wb') as f:
            pickle.dump(boundary_points_stack_euler, f)

        with open(os.path.join(vis_save_path, 'boundary_points_stack_euler_pure_resample.pkl'), 'wb') as f:
            pickle.dump(pure_resample, f)

        _plot_boundary()

        """
        20260113 added for test:
        
        """
        # output_dir = os.path.join(vis_save_path,"GVF_anomaly_detection_test")
        # os.makedirs(output_dir, exist_ok=True)
        # boundary_points_stack_euler_at, pure_resample_at = boundary_propagation_with_anomaly_detection(
        #                                                 displacement_fields_stack, 
        #                                                 rigid_registered_mask,
        #                                                 igl_unwarped_boundary_stack, 
        #                                                 output_dir,
        #                                                 window_size = 6,      
        #                                                 sigma_thresh = 3.0,  
        #                                                 min_history = 5,       
        #                                                 contour_distance_thres = 3.0,
        #                                                 stats_plot = True)
        # with open(os.path.join(vis_save_path, 'boundary_points_stack_euler_at.pkl'), 'wb') as f:
        #     pickle.dump(boundary_points_stack_euler_at, f)

        # with open(os.path.join(vis_save_path, 'boundary_points_stack_euler_pure_resample_at.pkl'), 'wb') as f:
        #     pickle.dump(pure_resample_at, f)

        # save_name = f"forward_euler_after_rotation_registration_pure_resample_anomaly_detection"
        # save_folder = os.path.join(vis_save_path, "boundary_visualization")
        # os.makedirs(save_folder, exist_ok=True)
        # save_matplot_video(registered_img_stack, pure_resample, save_folder, save_name, colormap='gray',img_plot=True,scatter = False, plot_multiple_point = True)

        # save_name = f"forward_euler_after_rotation_registration_100iter_lambda05_scatter_anomaly_detection"
        # save_folder = os.path.join(vis_save_path, "boundary_visualization")
        # os.makedirs(save_folder, exist_ok=True)
        # save_matplot_video(registered_img_stack, boundary_points_stack_euler, save_folder, save_name, colormap='gray',img_plot=True,scatter = False, plot_multiple_point = True)


    
    return boundary_points_stack_euler, mesh_2D_submesh_stack, igl_unwarped_boundary_stack

def interpolate_uv(first_frame_mesh_2D_submesh, igl_unwarped_boundary_stack, boundary_points_stack_euler, vis_save_path):
    mesh_2D_submesh, disk_coords, bdy_uv_0, bdy_index = generate_bdy_uv(first_frame_mesh_2D_submesh)
    igl_unwarped_reordered_boundary_stack = reorder_igl_boundary_stack(igl_unwarped_boundary_stack, boundary_points_stack_euler)
    bdy_uv_stack = interpolate_2nn_stack(boundary_points_stack_euler, igl_unwarped_reordered_boundary_stack, bdy_uv_0)

    save_name = f"interpolated_uv_vis"
    save_folder = os.path.join(vis_save_path, "uv_interpolation")
    os.makedirs(save_folder, exist_ok=True)
    img = np.zeros((len(bdy_uv_stack), 2,2)) # this actually won'tb be plotted
    save_matplot_video(img, bdy_uv_stack, save_folder, save_name, colormap='gray',img_plot=False,scatter = False, plot_multiple_point = True)

    return bdy_uv_stack, igl_unwarped_reordered_boundary_stack

def run_unwrap2D_mapping(img_stack,
                mask_stack,
                bdy_uv_stack,
                igl_unwarped_reordered_boundary_stack, 
                mesh_2D_submesh_stack,
                vis_save_path,
                *extra_img_stacks,
                plot=True,
                raster_size=512
                ):
    import unwrap3D.Image_Functions.image as image_fn
    import unwrap3D.Unzipping.unzip as uzip
    save_path = os.path.join(vis_save_path, "unwrap2D_pickle")
    v_out_stack, f_steps_out_stack, v_img_out_stack, bdy_index_stack = parallel_unwrap2D(img_stack, mask_stack,\
                         bdy_uv_stack, igl_unwarped_reordered_boundary_stack, mesh_2D_submesh_stack, save_path = save_path)

    unwrap_params_stack, unwrap_img_stack, unwrap_mask_stack = parallel_resample_img(v_out_stack, v_img_out_stack, \
        f_steps_out_stack, img_stack, raster_size, save_path = save_path)

    if plot:

        img_save_path = os.path.join(vis_save_path,"unwrap2D_img")
        vmin = np.nanmin(unwrap_img_stack[unwrap_img_stack is not None])
        vmax = np.nanmax(unwrap_img_stack[unwrap_img_stack is not None])
        os.makedirs(img_save_path,exist_ok=True)
        for ii, img in enumerate(unwrap_img_stack):
            if img is not None:

                fig, ax = plt.subplots(figsize=(10, 10))
                im = ax.imshow(img, cmap='magma', vmin=vmin, vmax=vmax)
                cbar = plt.colorbar(im, ax=ax)
                cbar.ax.tick_params(color='black', labelcolor='black')
                cbar.set_label('Intensity', color='black')
                cbar.outline.set_edgecolor('black')
                plt.savefig(os.path.join(img_save_path, f"frame_{ii+1}.tif"),
                            dpi=600, bbox_inches='tight')
                plt.close()

    extra_unwrap_stack = []
    for jj, extra_imgs in enumerate(extra_img_stacks):
        save_path = os.path.join(vis_save_path, f"unwrap2D_pickle_extra_{jj}")
        os.makedirs(save_path, exist_ok=True)
        unwrap_img_stack = []
        for img, unwrap_params in zip(extra_imgs, unwrap_params_stack):
            if unwrap_params is not None:
                unwrap_img = image_fn.map_intensity_interp2(unwrap_params.reshape(-1,2),
                                                    img.shape[:2], 
                                                    img,method='linear', fill_value=np.nan).reshape(unwrap_params.shape[:2])
                unwrap_img_stack.append(unwrap_img)
        unwrap_img_stack = np.array(unwrap_img_stack)
        extra_unwrap_stack.append(unwrap_img_stack)
        
        os.makedirs(save_path, exist_ok=True)
        save_unwrap_img   = os.path.join(save_path, 'unwrap_img_stack.pkl')
        save_unwrap_param = os.path.join(save_path, 'unwrap_params_stack.pkl')
        save_unwrap_mask  = os.path.join(save_path,'unwrap_mask_stack.pkl')

        with open(save_unwrap_img, 'wb') as f:
            pickle.dump(unwrap_img_stack, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(save_unwrap_param, 'wb') as f:
            pickle.dump(unwrap_params_stack, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(save_unwrap_mask, 'wb') as f:
            pickle.dump(unwrap_mask_stack, f, protocol=pickle.HIGHEST_PROTOCOL)

        if plot: 
    
            img_save_path = os.path.join(vis_save_path,f"unwrap2D_img_extra{jj}")
            os.makedirs(img_save_path, exist_ok=True)
          
            vmin = np.nanmin(unwrap_img_stack)
            vmax = np.nanmax(unwrap_img_stack)
    
            for ii, img in enumerate(unwrap_img_stack):
                if img is not None:
                    fig, ax = plt.subplots(figsize=(10, 10))

                    im = ax.imshow(img, cmap='magma', vmin=vmin, vmax=vmax)

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="4%", pad=0.05)

                    cbar = fig.colorbar(im, cax=cax)
                    cbar.ax.tick_params(color='black', labelcolor='black')
                    cbar.set_label('Intensity', color='black')
                    cbar.outline.set_edgecolor('black')

                    ax.set_axis_off()

                    fig.savefig(
                        os.path.join(img_save_path, f"frame_{ii+1}.tif"),
                        dpi=600
                    )
                    plt.close()
                # fig, ax = plt.subplots(figsize=(10, 10))
                # im = ax.imshow(img, cmap='magma', vmin=vmin, vmax=vmax)
                # cbar = plt.colorbar(im, ax=ax)
                # cbar.ax.tick_params(color='black', labelcolor='black')
                # cbar.set_label('Intensity', color='black')
                # cbar.outline.set_edgecolor('black')
                # plt.savefig(os.path.join(img_save_path, f"frame_{ii+1}.tif"),
                #             dpi=600, bbox_inches='tight')
                # plt.close()
    return (unwrap_params_stack, unwrap_img_stack, unwrap_mask_stack, *extra_unwrap_stack)


def run_unwrap2d_pipeline(
                    img_stack,
                    mask_stack,
                    vis_save_path,
                    rerun,
                    *extra_stacks,
                    extra_names=None,
                ):

    img_c, mask_c, *extra_c = step0_rigid_registration_center(
        img_stack, mask_stack, *extra_stacks, vis_save_path=os.path.join(vis_save_path, "Step0_rigid_registration_centered")
    )

    img_r, mask_r, *extra_r = step0_rotation_registration(
        img_c,
        mask_c, 
        vis_save_path, 
        rerun,
        *extra_c,
        extra_names=extra_names
    )

    img_c_2, mask_c_2, *extra_c_2 = step0_rigid_registration_center(
        img_r, mask_r, *extra_r, vis_save_path=os.path.join(vis_save_path, "Step0_2_rigid_registration_centered_again")
    )

    imwrite(os.path.join(vis_save_path, "Step0_2_rigid_registration_centered_again", "rigid_registered_cell.tif"), img_c_2)
    imwrite(os.path.join(vis_save_path, "Step0_2_rigid_registration_centered_again", "rigid_registered_mask.tif"), mask_c_2)
    for ii, img in enumerate(extra_c_2):
        imwrite(os.path.join(vis_save_path, "Step0_2_rigid_registration_centered_again", f"rigid_registered_extra{ii}.tif"), img)

    displacement_fields_stack = demon_results(mask_c_2, vis_save_path, rerun)
    boundary_points_stack_euler, mesh_2D_submesh_stack, igl_unwarped_boundary_stack = run_GVF(displacement_fields_stack, mask_c_2, img_c_2, vis_save_path, True)
    bdy_uv_stack, igl_unwarped_reordered_boundary_stack = interpolate_uv(mesh_2D_submesh_stack[0], igl_unwarped_boundary_stack, boundary_points_stack_euler,vis_save_path)
    unwrap_params_stack, unwrap_img_stack, unwrap_mask_stack, *extra_unwrap_stack = run_unwrap2D_mapping(img_c_2,
                mask_c_2, 
                bdy_uv_stack,
                igl_unwarped_reordered_boundary_stack, 
                mesh_2D_submesh_stack,
                vis_save_path,
                *extra_c_2
                )
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True,
                        help="Path to image stack (tif)")
    parser.add_argument("--mask", type=str, default=None,
                        help="Path to mask stack (tif)")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--extra", type=str, action="append", default=[],
            help="Additional image channels (repeatable)")
    parser.add_argument("--rerun", action="store_true",
            help="If some steps has already been run, skip this step if False; rerun if True (for demon registration and GVF)")


    args = parser.parse_args()

    img_stack = imread(args.img).astype(np.float32)  
    if args.mask is None:
        print("No mask provided, generating mask as img > 0")
        mask_stack = img_stack > 0

        mask_stack = np.array([
            remove_small_holes(m, area_threshold=200)
            for m in mask_stack
            ])

        mask_stack = np.array([
            remove_small_objects(m, min_size=50)
            for m in mask_stack
            ])

        # selem = disk(2)  

        # mask_stack = np.array([
        #     binary_closing(m, selem)
        #     for m in mask_stack
        #     ])

        # mask_stack = np.array([
        #     binary_opening(m, selem)
        #     for m in mask_stack
        #     ])

        mask_stack = mask_stack.astype(np.uint16)
    

        

        imwrite(os.path.join(args.out,"mask.tif"), mask_stack)
    else:
        mask_stack = imread(args.mask).astype(np.uint16)
  
    extra_stacks = [imread(p).astype(np.float32) for p in args.extra]
    
    run_unwrap2d_pipeline(
        img_stack,
        mask_stack,
        args.out,
        args.rerun,
        *extra_stacks,
        extra_names=["RhoA"]
    )



if __name__ == "__main__":
    main()
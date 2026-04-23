import multiprocessing as mp
import unwrap2D
import os 
import numpy as np
import pickle
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

unwrap_kwargs = dict(
    conformal_map=False,
    debugviz_tri_areadistort=False,
    areadistort_max_iter=100,
    areadistort_delta_h_bound=0.5,
    areadistort_stepsize=0.1,
    use_uniform_area_distort_relax=False,
    remesh_initial_mesh=False,
    average_edge_len_factor_initial=0.5,
    remesh_initial_conformal=False,
    average_edge_len_factor=1.0,
    area_distort_flip_tri=True,
    robust_L=True,
    eps=1,
    return_steps=False
)

def _init_child_mapping():
    global unwrap_2D
    import unwrap2D
    unwrap_2D = unwrap2D.unwrap_2D


def _worker_mapping(
    fr,
    img,
    mask,
    bdy_uv,
    reordered_igl_boundary,
    mesh_2D_submesh,
    conformal_map
):
    try:
        local_kwargs = dict(unwrap_kwargs)
        local_kwargs["conformal_map"] = conformal_map

        v_out, f_steps_out, submesh_vertices, bdy_index = unwrap_2D(
            img,
            mask,
            bnd_uv=bdy_uv,
            reordered_igl_boundary_fr=reordered_igl_boundary,
            mesh_2D_submesh=mesh_2D_submesh,
            **local_kwargs
        )
        return fr, v_out, f_steps_out, submesh_vertices, bdy_index, None

    except Exception as e:
        return fr, None, None, None, None, str(e)

    # v_out, f_steps_out, submesh_vertices, bdy_index = unwrap_2D(
    #     img,
    #     mask,
    #     bnd_uv = np.array(bdy_uv),
    #     reordered_igl_boundary_fr = reordered_igl_boundary,
    #     mesh_2D_submesh = mesh_2D_submesh,
    #     **_unwrap_kwargs
    # )
    # return fr, v_out, f_steps_out, submesh_vertices, bdy_index
def _worker_mapping_wrapper(args):
    return _worker_mapping(*args)

def parallel_unwrap2D(img_stack, mask_stack, bdy_uv_stack, reordered_igl_boundary_stack, mesh_2D_submesh_stack, save_path = None):
    N = len(img_stack)
    v_out_stack       = [None] * N
    f_steps_out_stack = [None] * N
    v_img_out_stack   = [None] * N
    bdy_index_stack   = [None] * N


    tasks = []
    for fr in range(N):
        img  = img_stack[fr]
        mask = mask_stack[fr]
        bdy_uv = bdy_uv_stack[fr]
        reordered_igl_boundary = reordered_igl_boundary_stack[fr]
        mesh_2D_submesh = mesh_2D_submesh_stack[fr]
        tasks.append((fr, img, mask, bdy_uv, reordered_igl_boundary, mesh_2D_submesh, False))


    ctx = mp.get_context("spawn")
    num_procs = min(mp.cpu_count(), 8) 

    TIMEOUT_SEC = 4 * 60

    with ctx.Pool(
            processes=num_procs,
            initializer=_init_child_mapping
        ) as pool:

        async_results = {}
        for task in tasks:
            fr = task[0]
            async_results[fr] = pool.apply_async(
                _worker_mapping_wrapper,
                args=(task,)
            )

        with tqdm(total=N, desc="Unwrapping") as pbar:
            for fr, async_res in async_results.items():
                try:
                    res = async_res.get(timeout=TIMEOUT_SEC)

                    fr, v_out, f_steps_out, submesh_vertices, bdy_index, err = res

                    if err is not None:
                        print(f"\n[Frame {fr}] Critical Error: {err}")
                    else:
                        v_out_stack[fr]       = v_out
                        f_steps_out_stack[fr] = f_steps_out
                        v_img_out_stack[fr]   = submesh_vertices
                        bdy_index_stack[fr]   = bdy_index
                    
                  
                except mp.TimeoutError:
                    print(f"[Frame {fr}] Timeout, retry with conformal_map")
                    fr0, img, mask, bdy_uv, reordered_igl_boundary, mesh_2D_submesh, _ = tasks[fr]
                    retry_task = (
                        fr0,
                        img,
                        mask,
                        bdy_uv,
                        reordered_igl_boundary,
                        mesh_2D_submesh,
                        True,  
                    )  

                    try:
                        retry_res = pool.apply_async(
                            _worker_mapping_wrapper,
                            args=(retry_task,)
                        ).get(timeout=TIMEOUT_SEC)

                        fr, v_out, f_steps_out, submesh_vertices, bdy_index, err = retry_res

                        if err is not None:
                            print(f"[Frame {fr}] Retry failed: {err}")
                        else:
                            v_out_stack[fr]       = v_out
                            f_steps_out_stack[fr] = f_steps_out
                            v_img_out_stack[fr]   = submesh_vertices
                            bdy_index_stack[fr]   = bdy_index

                    except mp.TimeoutError:
                        print(f"[Frame {fr}] Retry fails")

                    except Exception as e:
                        print(f"[Frame {fr}] Retry exception: {e}")

                except Exception as e:
                    print(f"[Frame {fr}] Exception (non-timeout): {e}")

                pbar.update(1)
                    

    

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        save_v_out_stack   = os.path.join(save_path, 'v_out_stack.pkl')
        save_f_steps_out_stack = os.path.join(save_path, 'f_steps_out_stack.pkl')
        save_v_img_out_stack  = os.path.join(save_path,'v_img_out_stack.pkl')
        save_bdy_index_stack  = os.path.join(save_path,'bdy_index_stack.pkl')

        with open(save_v_out_stack, 'wb') as f:
            pickle.dump(v_out_stack, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(save_f_steps_out_stack, 'wb') as f:
            pickle.dump(f_steps_out_stack, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(save_v_img_out_stack, 'wb') as f:
            pickle.dump(v_img_out_stack, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(save_bdy_index_stack, 'wb') as f:
            pickle.dump(bdy_index_stack, f, protocol=pickle.HIGHEST_PROTOCOL)
    return v_out_stack, f_steps_out_stack, v_img_out_stack, bdy_index_stack



def _init_child_resample():

    global resample_disk_grid_to_img_grid
    import unwrap2D as unwrap2D_fns
    resample_disk_grid_to_img_grid = unwrap2D_fns.resample_disk_grid_to_img_grid

def _worker_resample(args):
    fr, v_out, v_img_out, f_steps_out, raw_img, raster_size = args
    v_out_uv     = v_out[:, 1:]
    v_img_out_uv = v_img_out[:, 1:]
    unwrap_params, unwrap_img, unwrap_mask = resample_disk_grid_to_img_grid(
        v_out_uv,
        v_img_out_uv,     
        f_steps_out,
        raw_img,
        raster_size=raster_size,
        border_pad=32
    )
    return fr, unwrap_params, unwrap_img, unwrap_mask



def parallel_resample_img(v_out_stack, v_img_out_stack, f_steps_out_stack, img_stack, raster_size=256, save_path = None):
    frame_length = len(v_out_stack)
    valid_frames = [
        fr for fr in range(frame_length)
        if (
            v_out_stack[fr] is not None
            and v_img_out_stack[fr] is not None
            and f_steps_out_stack[fr] is not None
        )
    ]

    skipped_frames = sorted(set(range(frame_length)) - set(valid_frames))
    if len(skipped_frames) > 0:
        print(f"Resample Skipping {len(skipped_frames)} frames: {skipped_frames}")


    tasks = [
        (
            fr,
            v_out_stack[fr],
            v_img_out_stack[fr],
            f_steps_out_stack[fr],
            img_stack[fr],
            raster_size
        )
        for fr in valid_frames
    ]
    unwrap_params_stack = [None] * frame_length
    unwrap_img_stack    = [None] * frame_length
    unwrap_mask_stack   = [None] * frame_length


    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=8,                
        initializer=_init_child_resample,
        initargs=()
    ) as pool:
        
        for fr, unwrap_params, unwrap_img, unwrap_mask in tqdm(
            pool.imap(_worker_resample, tasks, chunksize=1),
            total=frame_length,
            desc="Resampling"
        ):
            unwrap_params_stack[fr] = unwrap_params
            unwrap_img_stack[fr]    = unwrap_img
            unwrap_mask_stack[fr]   = unwrap_mask

    if save_path is not None:
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
    return unwrap_params_stack, unwrap_img_stack, unwrap_mask_stack
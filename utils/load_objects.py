"""
This file contains several function to load previous-saved pickle files

"""
import os
import pickle

def load_uwnrap_objects(folder_path, suffix=''):
    

    save_v_out   = os.path.join(folder_path, f'v_out_stack{suffix}.pkl')
    save_v_img_out = os.path.join(folder_path, f'v_img_out_stack{suffix}.pkl')
    save_f_steps_out  = os.path.join(folder_path, f'f_steps_out_stack{suffix}.pkl')
    save_bdy_index  = os.path.join(folder_path, f'bdy_index_stack{suffix}.pkl')

    save_unwrap_img   = os.path.join(folder_path, f'unwrap_img_stack{suffix}.pkl')
    save_unwrap_param = os.path.join(folder_path, f'unwrap_params_stack{suffix}.pkl')
    save_unwrap_mask  = os.path.join(folder_path, f'unwrap_mask_stack{suffix}.pkl')

    with open(save_v_out, 'rb') as f:
        v_out_stack = pickle.load(f)
    with open(save_v_img_out, 'rb') as f:
        v_img_out_stack = pickle.load(f)
    with open(save_f_steps_out, 'rb') as f:
        f_steps_out_stack = pickle.load(f)
    with open(save_bdy_index, 'rb') as f:
        bdy_index_stack = pickle.load(f)

    with open(save_unwrap_img, 'rb') as f:
        unwrap_img_stack = pickle.load(f)
    with open(save_unwrap_param, 'rb') as f:
        unwrap_params_stack = pickle.load(f)
    with open(save_unwrap_mask, 'rb') as f:
        unwrap_mask_stack = pickle.load(f)

    return unwrap_img_stack, unwrap_params_stack, unwrap_mask_stack, v_out_stack, v_img_out_stack, f_steps_out_stack, bdy_index_stack



def load_demon_displacement(folder, file_name = 'displacement_fields_stack.pkl'):
    with open(os.path.join(save_path, file_name), 'rb') as f:
        displacement_fields_stack = pickle.load(f)

    return displacement_fields_stack

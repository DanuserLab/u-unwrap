import numpy as np
from scipy.io import loadmat
import os
import re


##### 
# u-Register window displacement calculation
#####

def load_cell_windows_layers(mat_path):
    windows = loadmat(mat_path)
    windows_layers = windows['windows'][0]
    lengths = [len(w[0]) for w in windows_layers]
    layer_num = max(lengths)
    return windows_layers, layer_num


def get_all_xys_mat_first_layers(mat_path,n_layers=1):

    windows_layers, layer_num = load_cell_windows_layers(mat_path)
    window_num = len(windows_layers)



    centroid_y = np.full((window_num, n_layers), np.nan, dtype=float)
    centroid_x = np.full((window_num, n_layers), np.nan, dtype=float)

    for theta_idx in range(window_num):
        first_layers = windows_layers[theta_idx]
        first_layers = first_layers[0]
  
        for ll in range(n_layers):
            
            if isinstance(first_layers, np.ndarray):
                try:
                    layers = first_layers[ll]
               
                    if len(layers) == 0:
                        # if no windows
                        continue
                    
                    layer_polys = layers[0]
                    lines = layer_polys

                    all_x, all_y = [], []
         
                    for line in lines:

                        x = line[0,:].ravel()
                        y = line[1,:].ravel()

                        all_x.append(x)
                        all_y.append(y)

                    if len(all_x)>0:
                        xcat = np.concatenate(all_x)
                        ycat = np.concatenate(all_y)
                        centroid_y[theta_idx, ll] = np.nanmean(ycat)
                        centroid_x[theta_idx, ll] = np.nanmean(xcat)
                except:
                    continue

    return centroid_y, centroid_x, window_num, layer_num



def cal_displacement(cy, cx,scale=1.0):
    
    cx = np.array(cx)
    cy = np.array(cy)
    dx = cx[1:] - cx[:-1]
    dy = cy[1:] - cy[:-1]
    dd = np.concatenate((dy[...,np.newaxis], dx[...,np.newaxis]),axis=-1)
    displacement = np.linalg.norm(dd, axis=2) * scale

    return displacement



def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]



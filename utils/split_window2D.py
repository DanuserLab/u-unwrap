import numpy as np
import skimage.segmentation as sksegmentation
import matplotlib.pyplot as plt
import os

SAVE_DIR = '/Users/yushiqiu/Desktop/UTSW/rotation_project/Gaudenz/track_window/output/2024_08_22'

class UnwrapWindow():
    def __init__(self, unwrap_img, unwrap_params, raster_size, border_pad):
        """
        A class to generate a window for the unwrapped image

        Input:
            unwrap_img: unwrapped image (as a disk form, but not in the disk scale)
            raster_size:  
            border_pad:
        """
        self.unwrap_img = unwrap_img
        self.unwrap_params = unwrap_params 
        self.raster_size = raster_size
        self.border_pad = border_pad

        self.center = (raster_size + 2 * border_pad) / 2.
        self.radius = raster_size / 2.

    def split_window(self, r_num, theta_num):
        """
        Split the unwrapped image into windows

        Input:
            r_num: the number of internal in r 
            theta_num: the number of internal in theta

        Output:
            segment_coords_uv: a list of the (v,u) index of the segments
            windows: corresponding values of the segments

        """
        segment_coords_uv, segment_coords_xy = self._generate_window_mask(r_num, theta_num)

        window_values = [self.unwrap_img[window[:,0],window[:,1]] for window in segment_coords_uv]

        return segment_coords_uv, segment_coords_xy, window_values
            
    def _generate_window_mask(self, r_num = 10, theta_num = 10, layer = 5):
        """
        Generate a mask for the disk

        Output:
            segment_coords_uv: a list of the (x,y) index of the segments in unwrap_img
        
        """
        self.theta_num = theta_num  
        self.r_num = r_num  

        h, w = self.unwrap_img.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        r = np.hypot(x - w // 2, y - h // 2)  
        theta = np.arctan2(y - h // 2, x - w // 2) 

        r_intervals = np.linspace(0, self.radius, num=r_num + 1)
        theta_intervals = np.linspace(-np.pi, np.pi, num=theta_num + 1) 

        segments = []
        segment_coords_uv = []
        segment_coords_xy = []
        for i in range(r_num-1, -1, -1):  ### windows are from the outer layer to the inner layer
            for j in range(theta_num):
                
                if i != r_num - 1:
                    r_mask = (r >= r_intervals[i]) & (r < r_intervals[i + 1])
                else:
                    r_mask = (r >= r_intervals[i]) & (r <= r_intervals[i + 1])
                
                if j != theta_num - 1:
                    theta_mask = (theta >= theta_intervals[j]) & (theta < theta_intervals[j + 1])
                else:
                    theta_mask = (theta >= theta_intervals[j]) & (theta <= theta_intervals[j + 1])
                    
                segment = np.logical_and(r_mask, theta_mask)
               
                segments.append(segment)
                
             

                uv = np.column_stack(np.where(segment))
    
                segment_coords_uv.append(uv)
                segment_coords_xy.append(self.unwrap_params[uv[:,0], uv[:,1]])

        self.segments = segments
        self.segments_coords_uv = segment_coords_uv
        self.segments_coords_xy = segment_coords_xy

        return segment_coords_uv, segment_coords_xy
    
    
    def plot_window(self,v_out,f_steps_out):
        
        plt.figure(figsize=(10, 10))
        plt.imshow(self.unwrap_img, cmap='gray')

        v_out_ = (v_out + 1)/2.* self.raster_size + self.border_pad
        plt.triplot(v_out_[:,2], 
                    v_out_[:,1],
                    f_steps_out, 
                    'g-', lw=0.3)

        for segment in self.segments:
            boundary = np.where(sksegmentation.find_boundaries(segment, mode='inner').astype(np.uint8))
            plt.scatter(boundary[1],boundary[0], c='r',s=0.1)
        plt.gca().set_aspect('auto')
        plt.savefig(os.path.join(SAVE_DIR, 'window.png'))
        plt.show()

    def get_layer_window(self, layer=5):
        edge_window_uv = []
        edge_window_xy = []
        for i in range(layer):
            layer_s = self.theta_num * (layer - 1)
            layer_e = self.theta_num * layer 
  
            uv = self.segments_coords_uv[layer_s:layer_e]
            xy = self.segments_coords_xy[layer_s:layer_e]

            edge_window_uv.append(uv)
            edge_window_xy.append(xy)

        return edge_window_uv, edge_window_xy
        
        


if __name__=="__main__":
    from matplotlib import cm 
    import os 
    import skimage.io as skio 
    import skimage.segmentation as sksegmentation
    import skimage.morphology as skmorph
    import skimage.measure as skmeasure
    import scipy.ndimage as ndimage


    import unwrap2D as unwrap2D_fns
    basefolder = '/Users/yushiqiu/Desktop/UTSW/rotation_project/Gaudenz/track_window'
    datafolder = os.path.join(basefolder,'data/Registration')
    savefolder = os.path.join(basefolder,'output/2024_08_15')

    mask = os.path.join(datafolder, 'MDACell5_Masks.tif')
    raw = os.path.join(datafolder, 'MDACell5_NoRegistration.tif')

    masks = skio.imread(mask)
    raw = skio.imread(raw)


    test_frame = 108

    mask_frame = masks[test_frame] > 0
    mask_frame = skmorph.binary_closing(mask_frame, skmorph.disk(1))
    mask_frame = ndimage.binary_fill_holes(mask_frame)

    v_out, f_steps_out, v_img_out, bdy_index = unwrap2D_fns.unwrap_2D(raw[test_frame], 
                                                            mask_frame, 
                                                            relax_tol=1.0e-5,
                                                            relax_niters=25,
                                                            relax_omega=1.,
                                                            debugviz_tri_areadistort = False,
                                                            areadistort_max_iter = 100,
                                                            areadistort_delta_h_bound = 0.5,
                                                            areadistort_stepsize=0.1,
                                                            area_distort_flip_tri=True)

    unwrap_params, unwrap_img, unwrap_mask = unwrap2D_fns.resample_disk_grid_to_img_grid(v_out[:,1:], 
                                                            v_img_out[:,1:], # use only the last two coordinates.
                                                            f_steps_out, 
                                                            raw[test_frame],
                                                            raster_size=256, border_pad=32) # raster_size is the diameter of the disk, border_pad=black area.


    
    unwrap_window = UnwrapWindow(unwrap_img, unwrap_params, 256, 32)
    segment_coords_uv, segment_coords_xy, window_values = unwrap_window.split_window(r_num=50, theta_num=100)
    
    """
    sanity check

    """
    count = 0
    for segment in segment_coords_uv:
        count += len(segment[0])
    a = unwrap_img>0
    print("count in segment_coords_uv: ", count)
    print("number of img", np.sum(a))
    
    unwrap_window.plot_window(v_out, f_steps_out)


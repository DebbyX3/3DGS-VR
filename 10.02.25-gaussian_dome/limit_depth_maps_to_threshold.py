import numpy as np
import os
import pylab as plt

read_from_folder = '../datasets/colmap_reconstructions/brg_rm_small_park-FullFrames/depth_after_fitting/exp_fit_da_non_metric_and_colmap_true_points'

for subdir, dirs, files in os.walk(read_from_folder):
    for file in files:

        filepath = subdir + os.sep + file

        if filepath.endswith(".npy"):

            depth_from_3DPoints = np.load(filepath)
            plt.imshow(depth_from_3DPoints, cmap='viridis')
            plt.title("Fitted Depth Map")
            plt.show()
            
            # Limit the depth map to a threshold
            threshold = 10
            depth_from_3DPoints[depth_from_3DPoints > threshold] = threshold

            # Create a binary mask using the new depth map  
            binary_mask = np.zeros_like(depth_from_3DPoints)
            binary_mask[depth_from_3DPoints > 0] = 1

            plt.imshow(binary_mask, cmap='viridis')
            plt.title("Binary Mask")
            plt.show()


            continue

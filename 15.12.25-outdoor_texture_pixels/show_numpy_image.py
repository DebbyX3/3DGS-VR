import numpy as np
import os
import pylab as plt

read_from_folder = '../datasets/colmap_reconstructions/brg_rm_small_park-FullFrames/depth_after_fitting/exp_fit_da_non_metric_and_colmap_true_points'

for subdir, dirs, files in os.walk(read_from_folder):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".npy"):

            depth_from_3DPoints = np.load(filepath)
            plt.imshow(depth_from_3DPoints, cmap='viridis')
            plt.title("Fitted Depth Map")
            plt.show()
            continue
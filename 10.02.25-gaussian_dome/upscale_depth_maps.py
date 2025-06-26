# important: use nearest neighbor interpolation for depth maps!!!

import cv2
import numpy as np
import os

input_folder = "../datasets/colmap_reconstructions/fields/video-depth-anything-metric/1280-720/20250402_175524_depths_npz"
output_folder = "../datasets/colmap_reconstructions/fields/video-depth-anything-metric/1920x1080/20250402_175524_depths_npz"

original_width = 1280
original_height = 720

resizeX = 1920
resizeY = 1080

os.makedirs(output_folder, exist_ok=True)

# FOR IMAGES
'''
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    depth = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    depth_up = cv2.resize(depth, (resizeX, resizeY), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(output_path, depth_up)
    print(f"Converted {filename}")
'''

# FOR NPZ FILES
for filename in os.listdir(input_folder):
    if filename.endswith(".npz"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        data = np.load(input_path)
        depth = data['depth']

        # Rimodella in 2D (altezza , larghezza)
        depth_reshape = depth.reshape((original_height, original_width))

        # Resize the depth map using nearest neighbor interpolation (larghezza, altezza)
        depth_up = cv2.resize(depth_reshape, (resizeX, resizeY), interpolation=cv2.INTER_NEAREST)

        # Save the resized depth map into a new npz file
        np.savez_compressed(output_path, depth=depth_up)
        print(f"Converted {filename}")
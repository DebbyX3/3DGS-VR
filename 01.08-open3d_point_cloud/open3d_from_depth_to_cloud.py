import open3d as o3d
import numpy as np
from PIL import Image

# Load the depth map image
depth_image = Image.open("path_to_depth_image.png")
depth_array = np.array(depth_image)

depth_array = np.load("../colmap_reconstructions/colmap_output_simple_radial/dense/stereo/depth_maps_npy_colmap/depth_map000001.png.geometric.bin.npy")


# Define camera intrinsics
fx = 500.0  # focal length in x-direction
fy = 500.0  # focal length in y-direction
cx = 320.0  # principal point x-coordinate
cy = 240.0  # principal point y-coordinate

intrinsic_matrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]])

# Create the point cloud from the depth map
point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_array, intrinsic_matrix)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])
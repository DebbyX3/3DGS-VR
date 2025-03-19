import open3d as o3d
import numpy as np

# Load colmap point cloud   
pcd = o3d.io.read_point_cloud("points3D_9subd_color.ply") 

o3d.visualization.draw_geometries([pcd])
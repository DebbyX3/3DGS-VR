import open3d as o3d
import numpy as np

# Load colmap point cloud   

pc = o3d.io.read_point_cloud("C:\\Users\\User\\Desktop\\Gaussian Splatting\\Video-Depth-Anything\\metric_depth\\pointcloudFields\\point0028.ply")


o3d.visualization.draw_geometries([pc])

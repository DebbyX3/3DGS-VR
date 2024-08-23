import open3d as o3d
import numpy as np

print("Load a ply point cloud, print it, and render it")

#ply_point_cloud = o3d.data.PLYPointCloud()
#pcd = o3d.io.read_point_cloud(ply_point_cloud.path)

pcd = o3d.io.read_point_cloud("input.ply")
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd])
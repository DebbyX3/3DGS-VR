import open3d as o3d
import numpy as np
from PIL import Image
import os
import matplotlib
matplotlib.use('TkAgg')
import pylab as plt
from typing import Union

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix using Hamilton convention."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / np.linalg.norm(q)  # Normalize quaternion

    # Estrarre i termini per maggiore chiarezza
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def extrinsics_from_quaternion_and_translation(qw, qx, qy, qz, tx, ty, tz):
    """Compute the extrinsic matrix [R | T] for a camera."""
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)

    # Translation vector
    T = np.array([tx, ty, tz]).reshape((3, 1))

    # Compute camera center C = -R^T * T
    #camera_center = -R.T @ T

    # Build the 4x4 extrinsic matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = T.flatten()  # Inseriamo T nella colonna finale

    return extrinsic

'''
The following function takes an Open3D PointCloud, equation of a plane (A, B, C, and D) 
and the optical center and returns a planar Open3D PointCloud Geometry.
'''
def get_flattened_pcds2(source,A,B,C,D,x0,y0,z0):
    x1 = np.asarray(source.points)[:,0]
    y1 = np.asarray(source.points)[:,1]
    z1 = np.asarray(source.points)[:,2]
    x0 = x0 * np.ones(x1.size)
    y0 = y0 * np.ones(y1.size)
    z0 = z0 * np.ones(z1.size)
    r = np.power(np.square(x1-x0)+np.square(y1-y0)+np.square(z1-z0),0.5)
    a = (x1-x0)/r
    b = (y1-y0)/r
    c = (z1-z0)/r
    t = -1 * (A * np.asarray(source.points)[:,0] + B * np.asarray(source.points)[:,1] + C * np.asarray(source.points)[:,2] + D)
    t = t / (a*A+b*B+c*C)
    np.asarray(source.points)[:,0] = x1 + a * t
    np.asarray(source.points)[:,1] = y1 + b * t
    np.asarray(source.points)[:,2] = z1 + c * t
    return source

def create_lines(mesh_vertices: np.ndarray, edges: o3d.utility.Vector2iVector, color: list = None) -> o3d.geometry.LineSet:
    """ Create a LineSet from vertices and edges of a mesh.
    :param mesh_vertices: vertices of the mesh
    :param edges: edges of the mesh, saved in a numpy array of shape (n, 2). Ex: [[0,5], [5,8], [8, 10]]
    :param color: color to apply to the edges, ex: GREEN, RED, ... (optional)
    :return: Lineset """

    print("mesh_vertices: ", mesh_vertices)

    # LineSet creation
    lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(mesh_vertices),
        lines=o3d.utility.Vector2iVector(np.array(edges)),
    )

    # Apply colors if exists, otherwise apply BLACK
    if color is not None:
        lines_color = [color] * len(edges)
    else:
        lines_color = [BLACK] * len(edges)
    lines.colors = o3d.utility.Vector3dVector(lines_color)

    return lines

def create_aabb(object_3d: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]) -> o3d.geometry.LineSet:
    """ Given a point cloud or a mesh this function computes the aabb of the 3d object
    :param object_3d: point cloud or mesh
    :return: aabb of the point cloud """

    vertices = np.array(object_3d.points)

    max_x = np.max(vertices[:, 0])  # max on x-axis
    max_y = np.max(vertices[:, 1])  # max on y-axis
    max_z = np.max(vertices[:, 2])  # max on z-axis

    min_x = np.min(vertices[:, 0])  # min on x-axis
    min_y = np.min(vertices[:, 1])  # min on y-axis
    min_z = np.min(vertices[:, 2])  # min on z-axis

    box_vertices = np.array([[max_x, max_y, min_z],
                             [max_x, max_y, max_z],
                             [min_x, max_y, max_z],
                             [min_x, max_y, min_z],
                             [max_x, min_y, min_z],
                             [max_x, min_y, max_z],
                             [min_x, min_y, max_z],
                             [min_x, min_y, min_z]])

    box_edges = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0],
                                            [4, 5], [5, 6], [6, 7], [7, 4],
                                            [0, 4], [1, 5], [2, 6], [3, 7]])

    aabb = create_lines(box_vertices, box_edges, [1,0,0])

    return aabb

# ************************** EXTRACT INTRINSICS FROM CAMERA.TXT FILE **************************
# Intrinsics matrix:
# [fx, 0, cx]
# [0, fy, cy]
# [0, 0,  1 ]


# ************** READ COLMAP CAMERA.TXT FILE    

# ***  WARNING: THIS SCRIPT ASSUMES THAT ALL CAMERAS HAVE THE SAME INTRINSICS ***
# ***  SO IN THE CAMERA.TXT FILE WE WILL ONLY READ THE FIRST CAMERA INTRINSICS ***
# *** (ALSO BEACUSE THERE IS ONLY ONE CAMERA IN THE CAMERA.TXT FILE IF THEY SHARE THE SAME INTRINSICS) ***

# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
#
# In case of Pinhole camera model (example):
# 1 PINHOLE 3072 2304 2560.56 2560.56 1536 1152
# 
# In case of Simple Pinhole camera model (example):
# 2 SIMPLE_PINHOLE 3072 2304 2559.81 1536 1152
#
# In case of Simple Radial camera model (example):
# 3 SIMPLE_RADIAL 3072 2304 2559.69 1536 1152 -0.0218531

camera_info = []

# Load the camera intrinsics 
with open('../colmap_reconstructions/water_bottle_gui_pinhole_1camera/sparse/cameras.txt', 'r') as f:
    for line in f:    
        # Ignore comments
        if not line.startswith("#"):
            single_camera_info = line.split() # split every field in line
            camera_info.append(single_camera_info) # and store them as separate fields as list in a list ( [ [] ] )

            # Camera info contains:
            # CAMERA_ID  MODEL   WIDTH   HEIGHT  PARAMS[]
            # 0          1       2       3       4   5   6   7   8
            # Where PARAMS[] are:
            # SIMPLE_PINHOLE: fx (fx = fy), cx, cy      1 focal length and principal point
            # PINHOLE: fx, fy, cx, cy                   2 focal lenghts and principal point
            # SIMPLE_RADIAL: fx (fx = fy), cx, cy, k1   1 focal length, principal point and radial distortion
            # RADIAL: fx (fx = fy), cx, cy, k1, k2      1 focal lengths, principal point and 2 radial distortions

            width = int(single_camera_info[2])
            height = int(single_camera_info[3])

            if single_camera_info[1] == "SIMPLE_PINHOLE":
                fx = float(single_camera_info[4])
                fy = float(single_camera_info[4]) #same as fx
                cx = float(single_camera_info[5])
                cy = float(single_camera_info[6])

            if single_camera_info[1] == "PINHOLE":
                fx = float(single_camera_info[4])
                fy = float(single_camera_info[5]) 
                cx = float(single_camera_info[6])
                cy = float(single_camera_info[7])

            if single_camera_info[1] == "SIMPLE_RADIAL":
                fx = float(single_camera_info[4])
                fy = float(single_camera_info[4]) #same as fx
                cx = float(single_camera_info[5])
                cy = float(single_camera_info[6])
                k1 = float(single_camera_info[7])

            if single_camera_info[1] == "RADIAL":
                fx = float(single_camera_info[4])
                fy = float(single_camera_info[4]) #same as fx
                cx = float(single_camera_info[5])
                cy = float(single_camera_info[6])
                k1 = float(single_camera_info[7])
                k2 = float(single_camera_info[8])  

            break    # We only need the first camera intrinsics (assume all cameras have the same intrinsics)  

# Create the camera intrinsic matrix
intrinsic_matrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]])

print("--- Camera: ", camera_info[0][1])
print(" Width: ", width)
print(" Height: ", height)
print(" fx: ", fx)
print(" fy: ", fy)
print(" cx: ", cx)
print(" cy: ", cy)  

if 'k1' in locals():
    print(" k1: ", k1)
if 'k2' in locals():
    print(" k2: ", k2)

#intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy) #alternatively
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_matrix)

count = 0
cameras_info = []
cameras_extrinsics = []

imgs_folder = "../colmap_reconstructions/water_bottle_gui_pinhole_1camera/images"
depth_map_folder = '../colmap_reconstructions/water_bottle_gui_pinhole_1camera/stereo/depth_maps'

point_cloud = o3d.geometry.PointCloud()

# ************************** EXTRACT EXTRINSICS FROM IMAGES.TXT FILE **************************
# Extrinsic matrix:
# [r1.1, r1.2, r1.3, tx]
# [r2.1, r2.2, r2.3, ty]
# [r3.1, r3.2, r3.3, tz]
# [0,    0,    0,    1 ]

with open('../colmap_reconstructions/water_bottle_gui_pinhole_1camera/sparse/images.txt', 'r') as f:
    for line in f:    
        # Ignore comments
        if not line.startswith("#"):
            count+=1
            #print(count)

            if(count > 0):

                if count % 2 != 0: # Read every other line (skip the second line for every image)
                    single_camera_info = line.split() # split every field in line
                    cameras_info.append(single_camera_info) # and store them as separate fields as list in a list ( [ [] ] )

                    # Images info contains:
                    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                    # 0         1   2   3   4   5   6   7   8          9

                    # CREATE ROTATION MATRIX 'R' FROM QUATERNIONS
                    quaternions = np.array([single_camera_info[1], single_camera_info[2], single_camera_info[3], single_camera_info[4]]) # numpy array
                    rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternions)

                    # CREATE TRANSLATION VECTOR T
                    translation = np.array([single_camera_info[5], single_camera_info[6], single_camera_info[7]], dtype = float)

                    # CREATE EXTRINSICS MATRIX                
                    extrinsics_matrix = np.vstack([np.hstack([rotation_matrix, translation.reshape(3, 1)]), 
                                                    np.array([0, 0, 0, 1])])

                    cameras_extrinsics.append(extrinsics_matrix)

                    #alternatively, to create the extrinsics
                    '''
                    #(qw, qx, qy, qz, tx, ty, tz):
                    extrinsics_matrix = extrinsics_from_quaternion_and_translation(single_camera_info[1], single_camera_info[2], single_camera_info[3], single_camera_info[4], single_camera_info[5], single_camera_info[6], single_camera_info[7])

                    print("new Extrinsic matrix:\n", extrinsics_matrix)
                    '''
                    
                    # Take the image file name
                    img_filename = single_camera_info[9]

                    # Read the depth map
                    depth_map_filename = img_filename + '.geometric.bin' # get the filename of the depth map
                    depth_map_path = os.path.join(depth_map_folder, depth_map_filename)
                    
                    depth = o3d.geometry.Image(read_array(depth_map_path)) 

                    # Read the image
                    img_path = os.path.join(imgs_folder, img_filename)
                    img = o3d.io.read_image(img_path)
                    rgb = o3d.geometry.Image(img)
                    
                    # Create the RGBD image
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_trunc=1000.0, depth_scale = 1.0, convert_rgb_to_intensity=False)

                    '''
                    # Plot both image and depth map
                    plt.subplot(1, 2, 1)
                    plt.title('Redwood grayscale image')
                    plt.imshow(rgbd.color)
                    plt.subplot(1, 2, 2)
                    plt.title('Redwood depth image')
                    plt.imshow(rgbd.depth, cmap='plasma')
                    plt.show()
                    '''

                    # Create the point cloud
                    #point_cloud += o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsics_matrix)

                    current_point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsics_matrix)

                    '''
                    The following function takes an Open3D PointCloud, equation of a plane (A, B, C, and D) 
                    and the optical center and returns a planar Open3D PointCloud Geometry.
                    z = 0
                    '''
                    flat = get_flattened_pcds2(current_point_cloud, 0, 0, 1, 0, 0, 0, -4)
                    flat.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) #flip it

                    x_max = np.max(np.asarray(flat.points)[:,0])
                    y_max = np.max(np.asarray(flat.points)[:,1])
                    z_max = np.max(np.asarray(flat.points)[:,2])

                    x_min = np.min(np.asarray(flat.points)[:,0])
                    y_min = np.min(np.asarray(flat.points)[:,1])
                    z_min = np.min(np.asarray(flat.points)[:,2])

                    print("max_point: ", x_max, y_max, z_max)
                    print("min_point: ", x_min, y_min, z_min)

                    max_point = np.array([x_max, y_max, z_max])
                    min_point = np.array([x_min, y_min, z_min])

                    groda1 = np.array([x_max, y_min, z_max])
                    groda2 = np.array([x_min, y_max, z_max])

                    print("max_point: ", max_point)
                    print("min_point: ", min_point)                    

                    for i in range(0, 3):
                        print(flat.points[i])

                    groda = flat.get_axis_aligned_bounding_box()
                
                    #lines = create_lines(np.array([[1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0]]), np.array([[0,1],[1,3],[3,2],[2,0]]), [1,0,0])
                    #lines = create_lines(np.array([first, [1,-1,0],[-1,1,0],[-1,-1,0]]), np.array([[0,1],[1,3],[3,2],[2,0]]), [1,0,0])
                    #lines = create_lines(np.array([groda1, max_point, groda2, min_point]), np.array([[0,1],[1,2],[2,3],[3,0]]), [1,0,0])

                    lines = create_aabb(flat)
                    
                    o3d.visualization.draw_geometries([flat, lines, groda])

            if count == 2:
                break

# Flip it, otherwise the pointcloud will be upside down
point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

#o3d.visualization.draw_geometries([point_cloud])

'''
#Save pointcloud to file
save_filename = "open3d_dense_pointcloud_water_bottle_gui_pinhole_1camera.ply"
o3d.io.write_point_cloud(save_filename, point_cloud, compressed = True)
'''
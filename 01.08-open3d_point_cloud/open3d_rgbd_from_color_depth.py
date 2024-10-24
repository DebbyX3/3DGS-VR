import open3d as o3d
import numpy as np
from PIL import Image
import os
import matplotlib
matplotlib.use('TkAgg')
import pylab as plt

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

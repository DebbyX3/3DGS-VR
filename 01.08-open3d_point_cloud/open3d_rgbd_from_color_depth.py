import open3d as o3d
import numpy as np
from PIL import Image
import os
import matplotlib
matplotlib.use('TkAgg')
import pylab as plt
from typing import Union
import time
import random

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

'''
equation of a sphere:
(x-h)^2 + (y-k)^2 + (z-l)^2 = r^2

The following python method takes an Open3D PointCloud geometry, 
and the radius (r1) and center (x0,y0,z0) of the sphere to project on.
'''
'''
def get_spherical_pcd(source,x0,y0,z0,r1):
    x2 = np.asarray(source.points)[:,0]
    y2 = np.asarray(source.points)[:,1]
    z2 = np.asarray(source.points)[:,2]
    x0 = x0 * np.ones(x2.size)
    y0 = y0 * np.ones(y2.size)
    z0 = z0 * np.ones(z2.size)
    r1 = r1 * np.ones(x2.size)
    r2 = np.power(np.square(x2-x0)+np.square(y2-y0)+np.square(z2-z0),0.5)
    np.asarray(source.points)[:,0] = x0 + (r1/r2) * (x2-x0)
    np.asarray(source.points)[:,1] = y0 + (r1/r2) * (y2-y0)
    np.asarray(source.points)[:,2] = z0 + (r1/r2) * (z2-z0)
    return source
'''

'''
The following python method takes an Open3D PointCloud geometry, 
and the radius (r1) and center (x0,y0,z0) of the sphere to project on.
'''
def createSphericalPointCloud(source_point_cloud, radius, xc, yc, zc):
    # For each point in the point cloud, find the intersection with the sphere
    # Practically, draw the line between the center of the sphere and the point in the point cloud, then find the intersection with the sphere

    outputpc = o3d.geometry.PointCloud()

    # Find the direction vector (a, b, c) between the center (C) of the sphere and the point (P) on the point cloud
    # to compute the equation of the line
    # P - C = (xp - xc, yp - yc, zp - zc) = (a, b, c) (and NOT C - P)
    directionVector = source_point_cloud.points - np.array([xc, yc, zc])

    a = directionVector[:,0]
    b = directionVector[:,1]
    c = directionVector[:,2]

    # Equation of the line (vector form)
    # (x,y,z)=(xC,yC,zC)+t(a,b,c)

    # Equation of the line (parametric form)
    # X = xc + t*a
    # Y = yc + t*b
    # Z = zc + t*c 

    # Sph eq: (x-h)^2 + (y-k)^2 + (z-l)^2 = r^2
    # Where (h, k, l) is the center (C) of the sphere and r is the radius

    # Substitute the equation of the line (X, Y, Z) into the equation of the sphere:
    # (xc + t*a - xc)^2 + (yc + t*b - yc)^2 + (zc + t*c - zc)^2 = r^2
    # = (t*a)^2 + (t*b)^2 + (t*c)^2 = r^2

    # Find t
    # t = +- (r / sqrt(a^2 + b^2 + c^2))

    # Take only the positive value of t, and substitute it back into the equation of the line to find the intersection point (X, Y, Z)
    # X = xc + a * (r / sqrt(a^2 + b^2 + c^2))
    # Y = yc + b * (r / sqrt(a^2 + b^2 + c^2))
    # Z = zc + c * (r / sqrt(a^2 + b^2 + c^2))

    # This is the final form that is written in the code below :)
    # The above steps are just for understanding

    

    t = (radius / np.sqrt(a**2 + b**2 + c**2)) 
    
    intersecX = xc + a * t
    intersecY = yc + b * t
    intersecZ = zc + c * t
    
    outputpc.points = o3d.utility.Vector3dVector(np.column_stack((intersecX, intersecY, intersecZ)))
    outputpc.colors = source_point_cloud.colors

    #comments to keep in mind: next, remove a point if the line magnitude is 0
    '''
    line_magnitude = np.sqrt(a**2 + b**2 + c**2)

    groda = np.where(line_magnitude == 0) # the line is a point! ignore it and delete from point cloud
    
    t = np.divide(radius, line_magnitude, out=1, where = line_magnitude != 0)

    # Remove the point with the known index
    known_index = 0  # Replace with the actual index you want to remove
    sourcepc.points = o3d.utility.Vector3dVector(np.delete(np.asarray(sourcepc.points), known_index, axis=0))
    sourcepc.colors = o3d.utility.Vector3dVector(np.delete(np.asarray(sourcepc.colors), known_index, axis=0))
    '''

    return outputpc

def createEquirectangularPointCloud(source_point_cloud, radius, xc, yc, zc):

    # Normalize coord so the sphere is centered in the center of the axis / scene
    x = np.asarray(source_point_cloud.points)[:,0] - xc
    y = np.asarray(source_point_cloud.points)[:,1] - yc
    z = np.asarray(source_point_cloud.points)[:,2] - zc
    
    radius = (np.sqrt(x**2 + y**2 + z**2)) # I already have it, but re-do the proper calculation just to be sure
    radius.astype(int)

    # Convert cartesian coordinates to spherical coordinates
    ratio = y / radius
    out_of_range = ratio[(ratio < -1) | (ratio > 1)]
    print("Valori fuori range:", out_of_range)

    # Polar coordinates
    theta = np.arctan2(z, x)
    phi = np.arcsin(y/radius)

    # Image size
    width =  2048
    height = 1024

    # Now that I have polar coords, convert to equirectangular image
    uCloud = (theta / (2 * np.pi)) * width              # from -pi to pi
    vCloud = (1 - (phi + np.pi / 2) / np.pi) * height   # from pi/2 to -pi/2

    '''
    # Let this commented:
    # This is the point cloud obtainted directly from the equirect calculations
    # It is not normalized, but I'm keeping it here for reference

    equirect_point_cloud = o3d.geometry.PointCloud()
    equirect_point_cloud.points = o3d.utility.Vector3dVector(np.column_stack((uCloud, vCloud, np.zeros(uCloud.size))))
    equirect_point_cloud.colors = sourcepc.colors
    '''

    # ---- Equirectangle Point Cloud
    # Normalize point U and V to be between 0 and 1
    uCloudNorm = (uCloud - np.min(uCloud)) / (np.max(uCloud) - np.min(uCloud))
    vCloudNorm = (vCloud - np.min(vCloud)) / (np.max(vCloud) - np.min(vCloud))

    # Then, multiply each dimension for width and height to stay in the image range
    uCloudNorm *= width
    vCloudNorm *= height

    # Fill a point cloud with the equirectangle normalized points
    normalized_equirect_point_cloud = o3d.geometry.PointCloud()
    normalized_equirect_point_cloud.points = o3d.utility.Vector3dVector(np.column_stack((uCloudNorm, vCloudNorm, np.zeros(uCloud.size))))
    normalized_equirect_point_cloud.colors = source_point_cloud.colors    

    # ---- Equirectangle Image
    # Just assign each cloud point color to the same pixel in the image
    # Create a new raster image
    img = np.zeros((height, width, 3), dtype=np.uint8) + 255 # White background
 
    # Assign colors to the raster image
    for i in range(len(normalized_equirect_point_cloud.points)):
        (x, y, z) = np.asarray(normalized_equirect_point_cloud.points[i], dtype=int)
        img[y-1, x-1] = (np.asarray(source_point_cloud.colors)[i] * 255).astype(np.uint8)

    #color_stack = np.zeros((height, width, 3), dtype=np.uint8)
    #una lista di array come 3 par, come faccio?
    
    '''
    # loop on altezza e lung
    for i in range(height):
        for k in range(width):
            groda = np.where(pointcloudnorm.points == [k, i, 0])
            color_stack[k, i] = np.asarray((pointcloudnorm.select_by_index(groda)).colors)
    '''

    '''
    # Visualizza l'immagine
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    '''

    return normalized_equirect_point_cloud

def pick_points(pcd):

    print("")
    print("")
    print("1) Please pick two points using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")    
    print("   Press [shift + +/-] to increase/decrease picked point size")
    print("2) After picking points, press q to close the window")
    print("")

    '''
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)  

    view_control = vis.get_view_control()
    view_control.set_constant_z_near(400)

    vis.run()  # user picks points
    vis.destroy_window()    
    '''

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    
    # Ottieni il controllo della vista
    view_control = vis.get_view_control()

    # Modifica i parametri della telecamera
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    # Avvicina la telecamera lungo l'asse Z
    extrinsic = np.array(camera_params.extrinsic, copy=True)

    # Stampa gli estrinseci iniziali
    print("Extrinsics prima:")
    print(extrinsic[2,3])

    extrinsic[2, 3] = 100  # Sposta la telecamera verso la scena lungo l'asse Z
    camera_params.extrinsic = extrinsic

    # Stampa gli estrinseci modificati
    print("Extrinsics dopo:")
    print(extrinsic[2,3])

    # Applica i nuovi parametri
    view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # Forza l'aggiornamento del rendering
    vis.poll_events()
    vis.update_renderer()


    # Esegui il visualizzatore
    vis.run()
    vis.destroy_window()
    


    '''
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)  

    dt = 0.2
    s = time.time()

    # run non-blocking visualization.
    keep_running = True
    while keep_running:

        if time.time() - s > dt:
            # workaround.
            # 1) Comment the 3 lines for the original behavior.
            # 2) Only comment the 1st and 3rd lines for reset only.
            # 3) Uncomment the 3 lines for the complete workaround.
            #cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
            vis.reset_view_point(True)
            #vis.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

        keep_running = vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()
    '''
    

    return vis.get_picked_points()

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

# ************************** PATHS **************************
cameraTxt_path = '../colmap_reconstructions/water_bottle_gui_pinhole_1camera/sparse/cameras.txt'
imagesTxt_path = '../colmap_reconstructions/water_bottle_gui_pinhole_1camera/sparse/images.txt'
imgs_folder = "../colmap_reconstructions/water_bottle_gui_pinhole_1camera/images"
depth_map_folder = '../colmap_reconstructions/water_bottle_gui_pinhole_1camera/stereo/depth_maps'

cameraTxt_path = '../colmap_reconstructions/colmap_output_simple_radial/sparse/cameras.txt'
imagesTxt_path = '../colmap_reconstructions/colmap_output_simple_radial/sparse/images.txt'
imgs_folder = "../colmap_reconstructions/colmap_output_simple_radial/dense/images"
depth_map_folder = '../colmap_reconstructions/colmap_output_simple_radial/dense/stereo/depth_maps'

cameraTxt_path = '../colmap_reconstructions/cavignal-bench_pinhole_1camera/sparse/cameras.txt'
imagesTxt_path = '../colmap_reconstructions/cavignal-bench_pinhole_1camera/sparse/images.txt'
imgs_folder = "../colmap_reconstructions/cavignal-bench_pinhole_1camera/dense/images"
depth_map_folder = '../colmap_reconstructions/cavignal-bench_pinhole_1camera/dense/stereo/depth_maps'

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
with open(cameraTxt_path, 'r') as f:
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

point_cloud = o3d.geometry.PointCloud()

# ************************** EXTRACT EXTRINSICS FROM IMAGES.TXT FILE **************************
# Extrinsic matrix:
# [r1.1, r1.2, r1.3, tx]
# [r2.1, r2.2, r2.3, ty]
# [r3.1, r3.2, r3.3, tz]
# [0,    0,    0,    1 ]

with open(imagesTxt_path, 'r') as f:
    for line in f:    
        # Ignore comments
        if not line.startswith("#"):
            count+=1

            if(count > 0):
                if count % 2 != 0: # Read every other line (skip the second line for every image)
                    if count % 19 == 0: # salta tot righe
                        
                        print(count)

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
                        
                        depth = read_array(depth_map_path)
                        max_depth = np.max(depth)
                        depth[depth == 0] = max_depth # points with 0 depth are set to the maximum depth value to mimic the infinity

                        depth = o3d.geometry.Image(depth) # convert to Open3D image

                        # Visualizza
                        #plt.imshow(depth)
                        #plt.axis('off')
                        #plt.show()


                        # Read the image
                        img_path = os.path.join(imgs_folder, img_filename)
                        img = o3d.io.read_image(img_path)
                        rgb = o3d.geometry.Image(img)
                        
                        # Create the RGBD image
                        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_trunc=1000.0, depth_scale = 1.0, convert_rgb_to_intensity=False)

                        # Create point cloud 
                        current_point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsics_matrix)

                        # Add to the total point cloud
                        point_cloud += current_point_cloud

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

                        '''
                        The following function takes an Open3D PointCloud, equation of a plane (A, B, C, and D) 
                        and the optical center and returns a planar Open3D PointCloud Geometry.
                        z = 0
                        '''
                        '''
                        flat = get_flattened_pcds2(current_point_cloud, 0, 0, 1, 0, 0, 0, -4)
                        flat.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) #flip it

                        x_max = np.max(np.asarray(flat.points)[:,0])
                        y_max = np.max(np.asarray(flat.points)[:,1])
                        z_max = np.max(np.asarray(flat.points)[:,2])

                        x_min = np.min(np.asarray(flat.points)[:,0])
                        y_min = np.min(np.asarray(flat.points)[:,1])
                        z_min = np.min(np.asarray(flat.points)[:,2])

                        max_point = np.array([x_max, y_max, z_max])
                        min_point = np.array([x_min, y_min, z_min])

                        groda1 = np.array([x_max, y_min, z_max])
                        groda2 = np.array([x_min, y_max, z_max])  

                        groda = flat.get_axis_aligned_bounding_box()
                        '''
                    
                        #lines = create_lines(np.array([[1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0]]), np.array([[0,1],[1,3],[3,2],[2,0]]), [1,0,0])
                        #lines = create_lines(np.array([first, [1,-1,0],[-1,1,0],[-1,-1,0]]), np.array([[0,1],[1,3],[3,2],[2,0]]), [1,0,0])
                        #lines = create_lines(np.array([groda1, max_point, groda2, min_point]), np.array([[0,1],[1,2],[2,3],[3,0]]), [1,0,0])

                        #lines = create_aabb(flat)
                        
                        '''
                        vis = o3d.visualization.Visualizer()
                        
                        vis.create_window()
                        vis.add_geometry(flat)
                        vis.poll_events()
                        vis.update_renderer()
                        vis.capture_screen_image("groda.jpg")
                        '''

                        #o3d.visualization.draw_geometries([flat, lines, groda])                      

            #if count >= 100:
            #    break




# Flip it, otherwise the point cloud will be upside down
point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points) * 1000) # test, fai la pc più grande

# **** SPHERIC PROJECTION

# PICK THE CENTER OF THE POINT CLOUD
data = []

while len(data) != 1:
    data = pick_points(point_cloud)
    if len(data) != 1:
        print("ERROR: please, pick exactly one center points!")
        input("Press Enter to retry...")

# Extract coordinates of the points
center_point_coords = point_cloud.select_by_index(data)

# Print points coordinates
print("Selected points coordinates: \n", np.asarray(center_point_coords.points))

# Convert to numpy array
center_coord = np.asarray(center_point_coords.points)[0]
print(center_coord[0], center_coord[1], center_coord[2])

# Remove points too close to the center
# ask the user a radius to remove the points in meters
sphere_radius = None

while not isinstance(sphere_radius, (int, float)):
    try:
        print("Input the radius of the sphere (in meters) that would crop the scene, leaving with only the outside of the sphere.")
        sphere_radius = float(input("   Use a 'period' to input a decimal number (e.g. 15.4): "))
    except ValueError:
        print("")
        print("ERROR: please, input a valid number!")
        print("")

# Crop points outside the sphere
distances_from_center = np.linalg.norm(np.asarray(point_cloud.points) - center_coord, axis=1)
indexes_points_outside_sphere = np.where(distances_from_center > sphere_radius)[0]

# Create a new point cloud with the points outside the sphere
cropped_point_cloud = point_cloud.select_by_index(indexes_points_outside_sphere)

# Visualize
#o3d.visualization.draw_geometries([cropped_point_cloud])

# Pass the new cropped point cloud and the center to the sphere function
sphere = createSphericalPointCloud(cropped_point_cloud, 5, center_coord[0], center_coord[1], center_coord[2])

# add center to a gimmick point cloud
# just to visualize the center
center_point_cloud = o3d.geometry.PointCloud()
center_point_cloud.points.append(center_coord)
#paint it magenta
center_point_cloud.colors.append(([1, 0, 1]))

o3d.visualization.draw_geometries([sphere, center_point_cloud])

equiImg = createEquirectangularPointCloud(sphere, 2, center_coord[0], center_coord[1], center_coord[2])
#print(equiImg)
o3d.visualization.draw_geometries([equiImg])

'''
#Save pointcloud to file
save_filename = "open3d_dense_pointcloud_water_bottle_gui_pinhole_1camera.ply"
o3d.io.write_point_cloud(save_filename, point_cloud, compressed = True)
'''
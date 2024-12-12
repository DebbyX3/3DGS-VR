import cv2
import open3d as o3d
import numpy as np
from PIL import Image
import os
import matplotlib
matplotlib.use('TkAgg')
import pylab as plt
import math
from collections import defaultdict

class EquirectangularInverseMapper:
    def __init__(self, output_height, output_width):
        """
        Initialize inverse mapper for equirectangular texture
        
        :param output_height: Height of output equirectangular texture
        :param output_width: Width of output equirectangular texture
        """
        self.height = output_height
        self.width = output_width
        
        # Pre-compute spherical coordinates for each texel
        u = np.linspace(0, 2*np.pi, output_width, endpoint=False)
        v = np.linspace(0, np.pi, output_height)
        self.U, self.V = np.meshgrid(u, v)
        
        # Convert to unit sphere coordinates
        self.X = np.sin(self.V) * np.cos(self.U)
        self.Y = np.sin(self.V) * np.sin(self.U)
        self.Z = np.cos(self.V)
    
    def project_to_camera(self, world_point, camera_matrix, dist_coeffs, rotation_matrix, translation_vector):
        """
        Project 3D world point to camera image coordinates
        
        :param world_point: 3D point in world coordinates
        :param camera_matrix: Camera intrinsic matrix
        :param rotation_matrix: Camera rotation matrix
        :param translation_vector: Camera translation vector
        :return: Projected pixel coordinates, depth
        """
        # Transform point to camera coordinates
        camera_point = rotation_matrix @ world_point + translation_vector
        
        # Project to image plane
        depth = np.linalg.norm(camera_point)
        
        # Use cv2 for precise projection
        projected_point, _ = cv2.projectPoints(
            world_point.reshape(1,1,3),
            cv2.Rodrigues(rotation_matrix)[0],
            translation_vector,
            camera_matrix,
            dist_coeffs
        )
        
        return projected_point[0,0], depth
    
    def inverse_map(self, cameras, fill_value=None):
        """
        Perform inverse mapping with z-buffer
        
        :param cameras: List of camera dictionaries
        :param fill_value: Value to use for unmapped texels (default: black)
        :return: Mapped equirectangular texture
        """
        # Initialize output texture and z-buffer
        if fill_value is None:
            fill_value = np.zeros(3, dtype=np.uint8)
        
        output_texture = np.full((self.height, self.width, 3), fill_value, dtype=np.uint8)
        z_buffer = np.full((self.height, self.width), np.inf)
        
        # Iterate through each texel
        for y in range(self.height):
            for x in range(self.width):
                # Get 3D world point for this texel
                world_point = np.array([
                    self.X[y, x],
                    self.Y[y, x],
                    self.Z[y, x]
                ])
                
                # Track best camera projection
                best_depth = np.inf
                best_color = fill_value
                
                # Check against each camera
                for camera in cameras:
                    try:
                        # Project point to camera
                        pixel, depth = self.project_to_camera(
                            world_point,
                            camera['intrinsic'],                            
                            camera.get('distortion', np.zeros(4)),
                            camera['rotation'],
                            camera['translation']
                        )
                        
                        # Check if pixel is inside image
                        cam_image = camera['image']
                        h, w = cam_image.shape[:2]
                        u, v = int(pixel[0]), int(pixel[1])
                        
                        if 0 <= u < w and 0 <= v < h:
                            # Z-buffer comparison
                            if depth < z_buffer[y, x]:
                                z_buffer[y, x] = depth
                                best_depth = depth
                                best_color = cam_image[v, u]
                    
                    except Exception as e:
                        print(f"Projection error: {e}")
                
                # Color the output texture
                output_texture[y, x] = best_color
        
        return output_texture

# Example usage
def main():

    cameraTxt_path = '../colmap_reconstructions/cavignal-bench_pinhole_1camera/sparse/cameras.txt'
    imagesTxt_path = '../colmap_reconstructions/cavignal-bench_pinhole_1camera/sparse/images.txt'
    imgs_folder = "../colmap_reconstructions/cavignal-bench_pinhole_1camera/dense/images"
    depth_map_folder = '../colmap_reconstructions/cavignal-bench_pinhole_1camera/dense/stereo/depth_maps'

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

    width, height = 2048, 1024

    # LINESET to draw camera directions in 3d as 'vectors'
    lineset = o3d.geometry.LineSet()
    all_points = []
    all_lines = []
    cameras_coords = []

    # LOOP INFO
    count = 0
    count_imgs = 0
    cameras_info = []
    cameras = []
    cameras_extrinsics = []
    # Read 1 image every 'skip'
    # e.g. If I have 10 imgs and skip = 3, read images:
    # 3, 6, 9
    skip = 6 # if 1: do not skip imgs
    print("-- You are reading 1 image every ", skip)

    with open(imagesTxt_path, 'r') as f:
        for line in f:    
            # Ignore comments
            if not line.startswith("#"):
                count+=1

                if(count > 0):
                    if count % 2 != 0: # Read every other line (skip the second line for every image)
                        count_imgs += 2
        
                        if count_imgs % skip == 0: # salta tot righe
                            
                            print("--- Img num ", (count_imgs/skip)/2 if skip %2 != 0 else count_imgs/skip)
                            print("Count: ", count)

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

                            # ----------- FIND CAMERA CENTER
                            # TRANSPOSE R (R^t)
                            rotation_transpose = rotation_matrix.transpose()

                            # MULTIPLY R_TRANSPOSED BY * (-1) (-R^t)
                            rotation_trans_inv = (-1) * rotation_transpose

                            # CREATE TRANSLATION VECTOR T
                            translation = np.array([single_camera_info[5], single_camera_info[6], single_camera_info[7]], dtype = float)

                            # DOT PRODUCT (*) BETWEEN INVERTED_R_TRANSPOSED (-R^t) AND TRANSLATION VECTOR (T)
                            # TO FIND CAMERA CENTER
                            camera_center = np.dot(rotation_trans_inv, translation)
                            
                            cameras_coords.append(camera_center)

                            # ------------ FIND CAMERA DIRECTION AND PREPARE TO DRAW IT AS A VECTOR IN 3D SPACE 

                            # Extract camera direction vector (forward vector)
                            # rotation_matrix = extrinsics_matrix[:3, :3]  # I already have the rot matrix, keep it commented
                            forward_vector = -rotation_matrix[:, 2]
                            
                            # cerca il punto finale per fare sta linea
                            # punto di inizio è la camera stessa
                            # punto finale = punto inizio + direzione * lunghezza vettore
                            final_point = camera_center + forward_vector * 1.5

                            # ora traccio linea
                            all_points.append(camera_center)
                            all_points.append(final_point)

                            # Aggiungi la linea tra gli ultimi due punti aggiunti
                            idx = len(all_points)
                            all_lines.append([idx - 2, idx - 1])  # Indici degli ultimi due punti

                            # Take the image file name
                            img_filename = single_camera_info[9]

                            # Read the image
                            img_path = os.path.join(imgs_folder, img_filename)
                            img = np.asarray(Image.open(img_path))

                            # ----- IMAGE TEXTURE INVERSE
                            cameras.append({'intrinsic': intrinsic_matrix, 
                                            'rotation': rotation_matrix, 
                                            'translation': translation, 
                                            'image': img})
    
    # Create inverse mapper for specific texture size
    mapper = EquirectangularInverseMapper(height, width)
    
    # Perform inverse mapping
    equirectangular_texture = mapper.inverse_map(cameras)
    
    # Save result
    #cv2.imwrite('generated_equirectangular.jpg', equirectangular_texture)

    plt.imshow(equirectangular_texture)
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.show()

if __name__ == '__main__':
    main()
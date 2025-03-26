import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

'''
NOTE: parse only the first camera since is's the same for all images.
In case of mutiple cameras, delete the 'break' and adjust the intrisics matrix to be a list of matrix
'''
def parse_cameras(camera_file_path):
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

    cameras_info = {}

    # Load the camera intrinsics 
    with open(camera_file_path, 'r') as f:
        for line in f:    
            # Ignore comments
            if not line.startswith("#"):
                single_camera_info = line.split() # split every field in line

                # Camera info contains:
                # CAMERA_ID  MODEL   WIDTH   HEIGHT  PARAMS[]
                # 0          1       2       3       4   5   6   7   8
                # Where PARAMS[] are:
                # SIMPLE_PINHOLE: fx (fx = fy), cx, cy      1 focal length and principal point
                # PINHOLE: fx, fy, cx, cy                   2 focal lenghts and principal point
                # SIMPLE_RADIAL: fx (fx = fy), cx, cy, k1   1 focal length, principal point and radial distortion
                # RADIAL: fx (fx = fy), cx, cy, k1, k2      1 focal lengths, principal point and 2 radial distortions

                camera_id = int(single_camera_info[0])

                camera_width = int(single_camera_info[2])
                camera_height = int(single_camera_info[3])

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

                cameras_info[camera_id] = {'type': single_camera_info[1], 'width': camera_width, 'height': camera_height, 'fx': fx, 'cx': cx, 'cy': cy}
                
                print("--- Camera: ", cameras_info[camera_id]['type'])
                print(" Width: ", camera_width)
                print(" Height: ", camera_height)
                print(" fx: ", fx)
                print(" fy: ", fy)
                print(" cx: ", cx)
                print(" cy: ", cy)  

                if 'k1' in locals():
                    print(" k1: ", k1)
                if 'k2' in locals():
                    print(" k2: ", k2)

                break    # We only need the first camera intrinsics (assume all cameras have the same intrinsics)  

    # Create the camera intrinsic matrix
    intrinsic_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])

    return cameras_info, intrinsic_matrix

def parse_images(images_file_path):
    """Parsa il file images.txt per estrarre le pose delle immagini e le corrispondenze 2D-3D."""
    # Images info contains two rows per image:

    #   1st row has:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   0         1   2   3   4   5   6   7   8          9

    #   2nd row has:
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    #   Example of this row:
    #   2362.39 248.498 58396       1784.7 268.254 59027        1784.7 268.254 -1
    #   X       Y       POINT3D_ID  X      Y       POINT3D_ID   X      Y       POINT3D_ID
    #   the last keypoint does not observe a 3D point in the reconstruction as the 3D point identifier is -1

    count = 0
    cameras_extrinsics = []
    images_info = {}

    with open(images_file_path, 'r') as f:

        for line in f:        
            # Ignore comments
            if not line.startswith("#"):
                count += 2

                if(count > 0):
                    #print("Image num: ", count/2)

                    # ---- FIRST LINE

                    img_first_line_info = line.split() # split every field in line

                    # Line contains:
                    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                    # 0         1   2   3   4   5   6   7   8          9

                    img_id = int(img_first_line_info[0])
                    qw, qx, qy, qz = map(float, img_first_line_info[1:5])
                    tx, ty, tz = map(float, img_first_line_info[5:8])
                    cam_id = int(img_first_line_info[8])
                    name = img_first_line_info[9]
                    
                    # CREATE ROTATION MATRIX 'R' FROM QUATERNIONS
                    rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
                    # quaternions = np.array([single_image_info[1], single_image_info[2], single_image_info[3], single_image_info[4]]) # numpy array
                    # rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternions)

                    # CREATE TRANSLATION VECTOR T
                    translation = np.array([tx, ty, tz], dtype = float).reshape(3, 1)
                    #translation = np.array([single_image_info[5], single_image_info[6], single_image_info[7]], dtype = float)

                    # CREATE EXTRINSICS MATRIX
                    extrinsic = np.hstack((rotation, translation))
                    extrinsic = np.vstack((extrinsic, [0, 0, 0, 1]))
                    #extrinsics_matrix = np.vstack([np.hstack([rotation_matrix, translation.reshape(3, 1)]), 
                    #                                np.array([0, 0, 0, 1])])

                    cameras_extrinsics.append(extrinsic)
                    
                    # ---- SECOND LINE
                    line = next(f)
                    img_second_line_info = line.split()

                    # Line contains:
                    # POINTS2D[] as (X, Y, POINT3D_ID)
                    #   Example of this row:
                    #   2362.39 248.498 58396       1784.7 268.254 59027        1784.7 268.254 -1
                    #   X       Y       POINT3D_ID  X      Y       POINT3D_ID   X      Y       POINT3D_ID

                    keypoints = []

                    # Process data in group of 3 (X, Y, POINT3D_ID)
                    point_data = list(map(float, img_second_line_info))
                    for j in range(0, len(point_data), 3):
                        px, py, point_id = point_data[j:j+3]
                        keypoints.append((px, py, int(point_id)))
                    
                    images_info[img_id] = {'name': name, 'camera_id': cam_id, 'extrinsic': extrinsic, 'keypoints': keypoints}

    print(f"Parsed {len(images_info)} images")
    return images_info

def parse_points3D(file_path):
    """Parsa il file points3D.txt per estrarre i punti 3D validi."""

    points3D = {}

    with open(file_path, 'r') as f:
        for line in f:

            if line.startswith('#') or not line.strip():
                continue

            parts = line.split()
            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            points3D[point_id] = np.array([x, y, z, 1.0])

    print(f"Parsed {len(points3D)} 3D points")
    return points3D

def generate_depth_map(image, camera, points3D):
    '''
    Estrae i parametri intrinseci della camera, inclusi la lunghezza focale (fx), il centro ottico (cx, cy).
    Inizializza la depth map come una matrice di zeri con le dimensioni dell'immagine.
    Itera sui keypoint dell'immagine, ignorando quelli senza un punto 3D valido.
    Trasforma i punti 3D nel sistema di riferimento della camera applicando la matrice extrinsic.
    Proietta i punti nella vista dell'immagine e calcola le coordinate pixel (x_proj, y_proj).
    Verifica i limiti dell'immagine e aggiorna la depth map con la distanza Z del punto 3D dalla camera.
    Restituisce la depth map e stampa un messaggio di debug.
    '''

    """Genera la depth map per una singola immagine."""
    width, height = camera['width'], camera['height']
    fx, cx, cy = camera['fx'], camera['cx'], camera['cy']
    intrinsic = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
    extrinsic = image['extrinsic']
    depth_map = np.zeros((height, width))
    
    for px, py, point_id in image['keypoints']:
        if point_id == -1 or point_id not in points3D:
            continue
        
        # Trasforma il punto 3D nel sistema di riferimento della camera
        point_cam = extrinsic @ points3D[point_id]
        if point_cam[2] <= 0:    # Ignora punti dietro la camera
            continue
        
        # Proietta il punto nello spazio dell'immagine
        x_proj = int(round((point_cam[0] * fx / point_cam[2]) + cx))
        y_proj = int(round((point_cam[1] * fx / point_cam[2]) + cy))
        
        # Verifica se il punto proiettato è all'interno dell'immagine
        if 0 <= x_proj < width and 0 <= y_proj < height:
            depth_map[y_proj, x_proj] = point_cam[2]
    
    print(f"Generated depth map for image {image['name']}")
    return depth_map

def main(colmap_folder):
    cameras_info, _  = parse_cameras(os.path.join(colmap_folder, 'cameras.txt'))
    images = parse_images(os.path.join(colmap_folder, 'images.txt'))
    points3D = parse_points3D(os.path.join(colmap_folder, 'points3D.txt'))
    
    depth_maps = {}    

    for img_id, img_data in images.items():
        camera = cameras_info[img_data['camera_id']]
        depth_map = generate_depth_map(img_data, camera, points3D)
        depth_maps[img_data['name']] = depth_map
        
        '''
        plt.imshow(depth_map, cmap='rainbow')
        plt.colorbar()
        plt.title(f"Depth Map: {img_data['name']}")
        plt.show()
        '''
    
    return depth_maps

# Esegui lo script
if __name__ == "__main__":
    colmap_folder = "../datasets/colmap_reconstructions/brg_rm_small_park-FullFrames/sparse"
    #colmap_folder = "../datasets/colmap_reconstructions/cavignal-fountain_pinhole_1camera/sparse/" 
    depth_maps = main(colmap_folder)

    for img_name, depth_map in depth_maps.items():
        np.save(os.path.join(colmap_folder, f"depth_maps_from_3DPoints/{img_name}_depth.npy"), depth_map)  # Salva ogni depth map come file NumPy
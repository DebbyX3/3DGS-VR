import open3d as o3d
import numpy as np
import struct
from plyfile import PlyData, PlyElement
import pycolmap
from PIL import Image
import collections

#### FROM COLMAP CODEBASE ####

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

class Image(BaseImage):
    def __new__(cls, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self = super(Image, cls).__new__(cls, id, qvec, tvec, camera_id, name, xys, point3D_ids)
        self._name = name
        return self

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
    
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)

def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")

def read_cameras_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras

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

#### END FROM COLMAP CODEBASE ####

def parse_cameras(cameras_bin_path):
    # ************** READ COLMAP CAMERA FILE    

    # ***  WARNING: THIS SCRIPT ASSUMES THAT ALL CAMERAS HAVE THE SAME INTRINSICS ***
    # ***  SO IN THE CAMERA FILE WE WILL ONLY READ THE FIRST CAMERA INTRINSICS ***
    # *** (ALSO BEACUSE THERE IS ONLY ONE CAMERA IN THE CAMERA FILE IF THEY SHARE THE SAME INTRINSICS) ***

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

    cameras_bin_info = read_cameras_binary(cameras_bin_path) 

    cameras_info = {}

    for cam_id, cam in cameras_bin_info.items():
        # Camera info contains:
        # CAMERA_ID  MODEL   WIDTH   HEIGHT  PARAMS[]
        # 0          1       2       3       4   5   6   7   8
        # Where PARAMS[] are:
        # SIMPLE_PINHOLE: fx (fx = fy), cx, cy      1 focal length and principal point
        # PINHOLE: fx, fy, cx, cy                   2 focal lenghts and principal point
        # SIMPLE_RADIAL: fx (fx = fy), cx, cy, k1   1 focal length, principal point and radial distortion
        # RADIAL: fx (fx = fy), cx, cy, k1, k2      1 focal lengths, principal point and 2 radial distortions

        if cam.model == "PINHOLE":
            cameras_info[cam_id] = {'id': cam_id, 
                                    'type': cam.model, 
                                    'width': cam.width, 
                                    'height': cam.height, 
                                    'fx': cam.params[0], 
                                    'fy': cam.params[1], 
                                    'cx': cam.params[2], 
                                    'cy': cam.params[3]}
        

        print("--- Camera ID: " + str(cameras_info[cam_id]['id']) + " - " + cameras_info[cam_id]['type'])
        print(" Width: ", cameras_info[cam_id]['width'])
        print(" Height: ", cameras_info[cam_id]['height'])
        print(" fx: ", cameras_info[cam_id]['fx'])
        print(" fy: ", cameras_info[cam_id]['fy'])
        print(" cx: ", cameras_info[cam_id]['cx'])
        print(" cy: ", cameras_info[cam_id]['cy'])  

        '''if 'k1' in locals():
            print(" k1: ", k1)
        if 'k2' in locals():
            print(" k2: ", k2)'''

        break    # We only need the first camera intrinsics (assume all cameras have the same intrinsics)  

    # Create the camera intrinsic matrix with the first (and only one) camera's intrinsics
    intrinsic_matrix = np.array([[cameras_info[1]['fx'],    0,                      cameras_info[1]['cx']],
                                [0,                         cameras_info[1]['fy'],  cameras_info[1]['cy']],
                                [0,                         0,                      1]])

    return cameras_info, intrinsic_matrix

def parse_images(images_bin_path):
    """Parsa il file images.bin per estrarre le pose delle immagini"""
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

    images_bin_info = read_images_binary(images_bin_path) 

    images_info = {}

    for img_id, img in images_bin_info.items():
        # COLMAP: ROTATION MATRIX 'R' FROM QUATERNION - FROM WORLD TO CAMERA
        qw, qx, qy, qz = img.qvec # please note that COLMAP writes/uses the quaternion in the order [qw, qx, qy, qz]
        rot_from_w_to_c = R.from_quat([qx, qy, qz, qw]).as_matrix() # but scipy uses the order [qx, qy, qz, qw] for quaternions

        # COLMAP: TRANSLATION VECTOR 'T' - FROM WORLD TO CAMERA
        tx, ty, tz = img.tvec
        trans_from_w_to_c = np.array([tx, ty, tz], dtype = float).reshape(3, 1)

        images_info[img_id] = {'id': img_id, 
                                'filename': img.name,
                                'rot_from_w_to_c': rot_from_w_to_c,
                                'trans_from_w_to_c': trans_from_w_to_c,
                                'camera_id': img.camera_id,
                                'points_2d': img.xys,
                                'point3D_ids': img.point3D_ids
                                }
    
        print("--- Image ID: " + str(images_info[img_id]['id']) + " - " + images_info[img_id]['filename'])
        
    return images_info




def batch_project_points_on_image(image, camera, points_3D):
    """
    Proietta in batch i punti 3D su un'immagine usando i parametri COLMAP (pycolmap) senza pycolmap.project_point().
    Restituisce:
        - point_ids_validi (lista di indici dei punti proiettati correttamente)
        - proiezioni 2D valide (array Nx2)
        - colori dei punti validi (array Nx3 o Nx4)
    """

    fx, fy, cx, cy = camera.params
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    # Matrice estrinseca
    qvec = image.rotation  # [qw, qx, qy, qz]
    tvec = image.translation  # [tx, ty, tz]

    R = qvec2rotmat(qvec)
    t = np.asarray(tvec).reshape(3, 1)

    # Punti in omogenee (Nx4)
    N = points_3D.shape[0]
    points_h = np.hstack((points_3D, np.ones((N, 1))))

    # Proiezione
    points_proj = (P @ points_h.T).T  # (N, 3)
    z = points_proj[:, 2]
    points_2D = points_proj[:, :2] / z[:, np.newaxis]  # omogenea -> euclidea

    # Filtro: davanti alla camera e dentro immagine
    valid_z = z > 0
    x, y = points_2D[:, 0], points_2D[:, 1]
    valid_x = (x >= 0) & (x < camera.width)
    valid_y = (y >= 0) & (y < camera.height)
    valid = valid_z & valid_x & valid_y

    valid_ids = np.where(valid)[0]
    valid_2D = points_2D[valid_ids]

    return valid_ids, valid_2D

def get_colors_from_image(image_path, pixel_coords):
    """
    Estrae i colori da una PIL image path in posizioni pixel_coords (Nx2).
    Salta i punti trasparenti.
    """
    image = Image.open(image_path)
    image_np = np.array(image)

    has_alpha = image_np.shape[2] == 4
    pixel_coords = np.round(pixel_coords).astype(int)
    x = pixel_coords[:, 0]
    y = pixel_coords[:, 1]

    # Proteggi da fuori bounds
    h, w = image_np.shape[:2]
    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x, y = x[in_bounds], y[in_bounds]
    colors = image_np[y, x]

    # Filtra trasparenti
    if has_alpha:
        visible_mask = colors[:, 3] > 0
        colors = colors[visible_mask]
        x, y = x[visible_mask], y[visible_mask]

    # Posizioni finali e colori
    positions_2D = np.stack([x, y], axis=1)
    return positions_2D, colors




def icosphere(subdivisions=2, radius=1.0, center=np.array([0.0, 0.0, 0.0]), return_centroids=False):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                         [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]])
    vertices /= np.linalg.norm(vertices[0])
    vertices *= radius
    
    faces = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                      [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    
    def midpoint(v1, v2):
        mid = (v1 + v2) / 2.0
        return mid / np.linalg.norm(mid) * radius
    
    for _ in range(subdivisions):
        new_faces = []
        midpoint_cache = {}
        
        def get_midpoint(i1, i2):
            if (i1, i2) not in midpoint_cache:
                if (i2, i1) in midpoint_cache:
                    return midpoint_cache[(i2, i1)]
                midpoint_cache[(i1, i2)] = len(vertices)
                vertices.append(midpoint(vertices[i1], vertices[i2]))
            return midpoint_cache[(i1, i2)]
        
        vertices = list(vertices)
        for f in faces:
            a, b, c = f
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        
        faces = np.array(new_faces)
        vertices = np.array(vertices)
    
    if return_centroids:
        centroids = np.mean(vertices[faces], axis=1)
        return centroids + center, faces
    
    return vertices + center, faces

def save_as_ply(points, colors, filename="gaussians.ply"):
    normals = -points / np.linalg.norm(points, axis=1, keepdims=True)  # Verso il centro
    vertex_data = np.array([
        (p[0], p[1], p[2], c[0], c[1], c[2], n[0], n[1], n[2])
        for p, c, n in zip(points, colors, normals)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u4'), ('green', 'u4'), ('blue', 'u4'),
               ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    ply_element = PlyElement.describe(vertex_data, 'vertex')
    PlyData([ply_element]).write(filename)

def find_most_distant_point(point_cloud, initial_point):
    # Convert point cloud to numpy array
    points = np.asarray(point_cloud.points)
    
    # Calculate the Euclidean distance from the initial point to all points in the point cloud
    distances = np.linalg.norm(points - initial_point, axis=1)
    
    # Find the index of the maximum distance
    max_distance_index = np.argmax(distances)
    
    # Get the most distant point
    most_distant_point = points[max_distance_index]
    
    # Get the maximum distance
    max_distance = distances[max_distance_index]
    
    return most_distant_point, max_distance

def calculate_circumradius(vertices, faces):
    circumradii = []
    for face in faces:
        a, b, c = vertices[face]
        # Lengths of sides of the triangle
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ca = np.linalg.norm(c - a)
        # Semi-perimeter
        s = (ab + bc + ca) / 2
        # Area of the triangle using Heron's formula
        area = np.sqrt(s * (s - ab) * (s - bc) * (s - ca))
        # Circumradius formula
        circumradius = (ab * bc * ca) / (4 * area)
        circumradii.append(circumradius)
    return max(circumradii)

def create_circle(center, normal, radius, resolution=30):
    """
    Create a circle in 3D space using Open3D.
    
    Parameters:
    - center: The center of the circle.
    - normal: The normal vector of the circle plane.
    - radius: The radius of the circle.
    - resolution: The number of points to generate the circle.
    
    Returns:
    - circle: An Open3D LineSet representing the circle.
    """
    theta = np.linspace(0, 2 * np.pi, resolution)
    circle_points = np.array([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)]).T
    
    # Create a rotation matrix to align the circle with the normal vector
    z_axis = np.array([0, 0, 1])
    normal = normal / np.linalg.norm(normal)
    v = np.cross(z_axis, normal)
    c = np.dot(z_axis, normal)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + k + k @ k * (1 / (1 + c))
    
    # Rotate and translate the circle points
    circle_points = circle_points @ rotation_matrix.T + center
    
    # Create the LineSet for the circle
    lines = [[i, (i + 1) % resolution] for i in range(resolution)]
    circle = o3d.geometry.LineSet()
    circle.points = o3d.utility.Vector3dVector(circle_points)
    circle.lines = o3d.utility.Vector2iVector(lines)
    return circle

# Funzione per calcolare la distanza euclidea tra due punti 3D
def distance_3d(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)




subdivisions = 9 #9 default
radius_mult = 4  # metti *4 o *5



# -- BrgRm small park folders
sparse_folder = '../datasets/colmap_reconstructions/brg_rm_small_park-FullFrames/sparse'
image_folder = '../datasets/colmap_reconstructions/brg_rm_small_park-FullFrames/images_threshold_10/'

# -- Field folders
#sparse_folder = '../datasets/colmap_reconstructions/fields/sparse/0/'
#image_folder = '../datasets/colmap_reconstructions/fields/input/'

# -- BrgRm small park complete pipeline folders
sparse_folder = '../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/sparse/0'
cameras_bin_path = "../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/sparse/0/cameras.bin"
images_bin_path = "../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/sparse/0/images.bin"
images_folder = "../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/images"
image_folder = '../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/video-depth-anything-metric/fromSceneCenter/distances_threshold_30.0'
save_ply_path = f'../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/points3D_{subdivisions}subd_{radius_mult}radius_color_test.ply'

reconstruction = pycolmap.Reconstruction(sparse_folder)
print(reconstruction.summary())

# LINESET to draw camera directions in 3d as 'vectors'
lineset = o3d.geometry.LineSet()
all_points = []
all_lines = []

cameras_info, intrinsic_matrix = parse_cameras(cameras_bin_path)
images_info = parse_images(images_bin_path)

# loop on all images
camera_data = {}

for image_id, image in reconstruction.images.items():
    camera_coords = image.projection_center()
    direction_vector = image.viewing_direction()

    # cerca il punto finale per fare sta linea
    # punto di inizio è la camera stessa
    # punto finale = punto inizio + direzione * lunghezza vettore
    final_point = camera_coords + direction_vector * 1.5

    # ora traccio linea
    all_points.append(camera_coords)
    all_points.append(final_point)

    # Aggiungi la linea tra gli ultimi due punti aggiunti
    idx = len(all_points)
    all_lines.append([idx - 2, idx - 1])  # Indici degli ultimi due punti

    # Store camera data
    camera_data[image_id] = {
        "center": camera_coords,
        "direction": direction_vector
    }

'''
# Accessing the values outside the loop
for camera_id, data in camera_data.items(): # camera_id = key, data = value
    print(f"Camera {camera_id} center: {data['center']}")
    print(f"Camera {camera_id} direction: {data['direction']}")
'''

# -------- Find center of all cameras (center of point cloud)

# Create new point cloud, add camera centers
cameras_point_cloud = o3d.geometry.PointCloud()
# extract from dict
cameras_coords = [data['center'] for data in camera_data.values()]
cameras_point_cloud.points = o3d.utility.Vector3dVector(cameras_coords)
center_of_scene = cameras_point_cloud.get_center()

most_distant_point, max_distance = find_most_distant_point(cameras_point_cloud, center_of_scene)
print("Most distant point:", most_distant_point)
print("Distance:", max_distance)
print("Center of scene:", center_of_scene)


# --------- create icosphere based on max_distance
# radius = max_distance * 2
ico_points_pos, ico_faces = icosphere(subdivisions = subdivisions, 
                            radius = max_distance * radius_mult,
                            center = center_of_scene, 
                            return_centroids = False) #keep false!!!!!!!!!!!

print("num of icosphere points generated: ", ico_points_pos.size/3)
print("radius of icosphere: ", max_distance * radius_mult)

icosphere_pc = o3d.geometry.PointCloud()
icosphere_pc.points = o3d.utility.Vector3dVector(ico_points_pos)
#attenzione! 255 sono rgb, o3d li vorrebbe 0-1, però visto che salvo gli rgb nel ply, devo tenere così con 255
icosphere_pc.colors = o3d.utility.Vector3dVector(np.full((len(ico_points_pos), 3), 255))

# ------------- project 3d point onto the image

point_colors = {}

for image_id, image in reconstruction.images.items():
    #image = reconstruction.images[image_id]

    image_path = image.name
    try:
        image_data = Image.open(image_folder + "/" + image_path)
    except:
        print("!!!! " + image_path + " not found")
        continue
    #print("Opened: " + image_folder + "/" + image_path)

    point_3D_id = 0

    image_path = image_folder + "/" + image.name
    point_coords = np.asarray(icosphere_pc.points)
    valid_ids, projected = batch_project_points_on_image(image, reconstruction.cameras[image.camera_id], point_coords)
    pixel_2D, colors = get_colors_from_image(image_path, projected)

    for i, (color, coord, pid) in enumerate(zip(colors, pixel_2D, valid_ids)):
        if pid not in point_colors:
            point_colors[pid] = []
        point_colors[pid].append((tuple(color), coord, point_coords[pid], pid))

# Alla fine, scegli il colore più vicino per ciascun punto 3D
for point_3D_id, infos in point_colors.items():
    # Calcola la distanza dal centro della scena per ciascun colore
    distances = [distance_3d(center_of_scene, color[2]) for color in infos]
    closest_index = np.argmin(distances)  # Trova l'indice del colore più vicino

    # Scegli il colore più vicino
    chosen_color = infos[closest_index][0]

    if len(chosen_color) >= 4:  # if the color has an alpha channel
        icosphere_pc.colors[point_3D_id] = chosen_color[:3] # remove the alpha channel and just save the rbg
    else: # if the color has NOT an alpha channel
        icosphere_pc.colors[point_3D_id] = chosen_color


    #print(f"Punto 3D {point_3D_id}: Colore scelto (più vicino al centro della scena): {chosen_color}")

print("finita proiezione colori\nInizio ricerca raggio ottimo")

# --------- find optimal radius of circles of point on the icosphere to cover all other circles
optimal_radius = calculate_circumradius(ico_points_pos, ico_faces)
print("Optimal radius for circles:", optimal_radius)

'''
# --------- Create circles on the icosphere points
circles = []

for point in ico_points_pos:
    circle = create_circle(center=point, normal=point - center_of_scene, radius=optimal_radius)
    circles.append(circle)
'''

# ------- SHOW CAMERAS IN 3D (RED) + FORWARD VECTOR (GREEN)
# Paint camera coords red
cameras_point_cloud.paint_uniform_color([1, 0, 0])

# Create lineset
lineset.points = o3d.utility.Vector3dVector(all_points)
lineset.lines = o3d.utility.Vector2iVector(all_lines)

# Apply color to lineset
GREEN = [0.0, 1.0, 0.0]
lines_color = [GREEN] * len(lineset.lines)
lineset.colors = o3d.utility.Vector3dVector(lines_color)


save_as_ply(ico_points_pos, icosphere_pc.colors, save_ply_path)


# SE VUOI VISUALIZZARE SU OPEN3D, DIVIDI TUTTI I COLORI PER 255
# NON LI SALVO 'DIVISI' PERCHE GS LI VUOLE COME COLMAP, CIOE RGB CLASSICI

test = np.asarray(icosphere_pc.colors)/255
icosphere_pc.colors = o3d.utility.Vector3dVector(test)

#o3d.visualization.draw_geometries([lineset, cameras_point_cloud, icosphere_pc] + circles)
o3d.visualization.draw_geometries([lineset, cameras_point_cloud, icosphere_pc])


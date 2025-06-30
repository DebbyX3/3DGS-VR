import open3d as o3d
import numpy as np
import struct
from plyfile import PlyData, PlyElement
import pycolmap
from PIL import Image

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




subdivisions = 2 #9 default
radius_mult = 4  # metti *4 o *5



# -- BrgRm small park folders
sparse_folder = '../datasets/colmap_reconstructions/brg_rm_small_park-FullFrames/sparse'
image_folder = '../datasets/colmap_reconstructions/brg_rm_small_park-FullFrames/images_threshold_10/'

# -- Field folders
#sparse_folder = '../datasets/colmap_reconstructions/fields/sparse/0/'
#image_folder = '../datasets/colmap_reconstructions/fields/input/'

# -- BrgRm small park complete pipeline folders
sparse_folder = '../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/sparse/0'
image_folder = '../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/video-depth-anything-metric/fromSceneCenter/distances_threshold_30.0'
save_ply_path = f'../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/points3D_{subdivisions}subd_{radius_mult}radius_color.ply'

reconstruction = pycolmap.Reconstruction(sparse_folder)
print(reconstruction.summary())

# LINESET to draw camera directions in 3d as 'vectors'
lineset = o3d.geometry.LineSet()
all_points = []
all_lines = []

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

    for point_3D in icosphere_pc.points:
        # Proietta il punto 3D nell'immagine
        point_2D = image.project_point(point_3D)

        # Verifica se il punto è stato proiettato correttamente
        if point_2D is not None:
            x_pixel, y_pixel = int(round(point_2D[0])), int(round(point_2D[1]))

            # Verifica se il pixel è dentro i limiti dell'immagine
            if 0 <= x_pixel < image_data.width and 0 <= y_pixel < image_data.height:                
                # Estrai il colore del pixel
                color = image_data.getpixel((x_pixel, y_pixel))

                # Check color:
                # - if color has an alpha channel AND it is NOT transparent (= 0)
                # OR
                # - if the color has NOT an alpha channel
                # Then save the color!
                # Basically, I do NOT want to save a trasnparent color
                if((len(color) > 3 and color[3] != 0) or len(color) <= 3): #short circuit evaluation without throwing an exceptions if color[3] does not exists
                    if point_3D_id not in point_colors:
                        point_colors[point_3D_id] = []
                    point_colors[point_3D_id].append((color, point_2D, point_3D, point_3D_id))  # Colore, posizione 2D e 3D

        point_3D_id += 1

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


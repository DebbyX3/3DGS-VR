# examples/Python/Advanced/interactive_visualization.py

import numpy as np
import open3d as o3d
import copy

def pick_points(pcd):

    print("")
    print("1) Please pick two points using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")    
    print("   Press [shift + +/-] to increase/decrease picked point size")
    print("2) After picking points, press q to close the window")
    print("")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()

    print("")

    return vis.get_picked_points()


if __name__ == "__main__":

    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_point_cloud.path)

    pcd = o3d.io.read_point_cloud('fused.ply')

    # ----------- PICK EXACTLY TWO POINTS IN THE POINT CLOUD AND INPUT THEIR REAL KNOWN DISTANCE (IN METERS)  ------------

    data = []

    while len(data) != 2:
        data = pick_points(pcd)
        if len(data) != 2:
            print("ERROR: please, pick exactly two points!")
            input("Press Enter to retry...")

    print("ID of selected points: \n", data)

    real_distance = None

    while not isinstance(real_distance, (int, float)):
        try:
            print("Input the REAL known distance between the two selected points in meters.")
            real_distance = float(input("   Use a 'period' to input a decimal number (e.g. 2.1): "))
        except ValueError:
            print("")
            print("ERROR: please, input a valid number!")
            print("")


    # ---------- COMPUTE THE MULTIPLIER FACTOR TO SCALE THE POINT CLOUD TO THE REAL DIMENSIONS AND SIZE  ------------

    # Extract coordinates of the points
    selected_points_coords = pcd.select_by_index(data)

    # Print points coordinates
    print("Selected points coordinates: \n", np.asarray(selected_points_coords.points))

    # Original distance between the two points in the point cloud
    original_distance = np.linalg.norm(np.asarray(selected_points_coords.points[0]) - np.asarray(selected_points_coords.points[1]))
    print("Original distance: ", original_distance)

    # Calculate multiplier factor to correctly scale the original point cloud to the real dimensions and size
    # real_distance / old_distance = mult_factor (e.g. 6 (real) / 2 (old) = 3)

    mult_factor = real_distance / original_distance
    print("Multiplier factor: ", mult_factor)

    # ---------- MULTIPLY EACH POINT OF THE POINT CLOUD TO THE FACTOR TO SCALE IT TO THE REAL SIZE  ------------

    # Multiply each point of the point cloud by the mult_factor to scale it to the real size

    # Save them in a new point cloud
    scaled_point_cloud = o3d.geometry.PointCloud()
    scaled_points = np.asarray(pcd.points) * mult_factor
    scaled_point_cloud.points = o3d.utility.Vector3dVector(scaled_points) #assign the new scaled points
    scaled_point_cloud.colors = o3d.utility.Vector3dVector(pcd.colors) #assign the same colors

    #scaled_point_cloud.paint_uniform_color([1, 0, 0]) #red 

    # New distance between the two points in the scaled point cloud

    # Extract coordinates of the points
    new_selected_points_coords = scaled_point_cloud.select_by_index(data)

    distance_check = np.linalg.norm(scaled_points[data[0]] - scaled_points[data[1]])
    print("New distance in the scaled point cloud between the same 2 points (just to double check): ", distance_check)

    #print("Selected points new coordinates - after scaling: \n", new_selected_points_coords)


    # center of the point cloud
    center = scaled_point_cloud.get_center()
    #color that point in green
    print("center:", center)
    

    # Find the nearest point to the 'center' point
    distances = np.linalg.norm(np.asarray(scaled_point_cloud.points) - np.asarray(center), axis=1)
    nearest_center_point_idx = np.argmin(distances)
    nearest_center_point = scaled_point_cloud.points[nearest_center_point_idx]
    print("Nearest point to the center:", nearest_center_point)

    # Change the color of the nearest center point in the point cloud 'scaled_point_cloud'
    colors = np.asarray(scaled_point_cloud.colors)
    colors[nearest_center_point_idx] = [0, 1, 0]  # RGB for green

    # add center to the point cloud
    scaled_point_cloud.points.append(center)
    #paint it magenta
    scaled_point_cloud.colors.append(([1, 0, 1]))

    # ask the user a radius for the sphere in meters
    sphere_radius = None

    while not isinstance(sphere_radius, (int, float)):
        try:
            print("Input the radius of the sphere (in meters) that would crop the scene, leaving with only the outside of the sphere.")
            sphere_radius = float(input("   Use a 'period' to input a decimal number (e.g. 15.4): "))
        except ValueError:
            print("")
            print("ERROR: please, input a valid number!")
            print("")

    # create a sphere at the center using the radius sphere_radius
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere_in_center = copy.deepcopy(sphere).translate(center, relative=False) #False, the center of the geometry is translated directly to the position specified in the first argument.

    mat_sphere = o3d.visualization.rendering.MaterialRecord()
    mat_sphere.shader = 'defaultLit'
    mat_sphere.base_color = [0.5, 0, 0, 0.5]

    sphere_complete = {'name': 'sphere', 'geometry': sphere_in_center, 'material': mat_sphere}

    # crop
    #cerca tutti i punti che sono a distanza minore di sphere_radius dal centro
    # quelli minori vanno eliminati
    distances_from_center = np.linalg.norm(np.asarray(scaled_point_cloud.points) - np.asarray(center), axis=1)
    indexes_points_outside_sphere = np.where(distances_from_center > sphere_radius)[0]

    # Save them in a new point cloud
    cropped_point_cloud = o3d.geometry.PointCloud()

    for idx in indexes_points_outside_sphere:
        cropped_point_cloud.points.append(scaled_point_cloud.points[idx])
        cropped_point_cloud.colors.append(scaled_point_cloud.colors[idx])
    
    # Visualize
    o3d.visualization.draw_geometries([cropped_point_cloud, sphere_in_center])

    ''' 
    # Color selected points in red
    colors = np.asarray(pcd.colors)

    for idx in data:
        colors[idx] = [1, 0, 0]  # RGB for red

    # Visualize the selected points
    o3d.visualization.draw_geometries([pcd])
    '''

    
import cv2
import numpy as np

def separate_normal_image(depth_map_path, normal_image_path, threshold):
    # Load the depth map image
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

    # Load the normal image
    normal_image = cv2.imread(normal_image_path)

    # Apply threshold to the depth map
    _, thresholded_depth_map = cv2.threshold(depth_map, threshold, 255, cv2.THRESH_BINARY)

    # Invert the thresholded depth map
    inverted_depth_map = cv2.bitwise_not(thresholded_depth_map)

    # Apply the inverted depth map as a mask to the normal image
    separated_image_1 = cv2.bitwise_and(normal_image, normal_image, mask=inverted_depth_map)
    separated_image_2 = cv2.bitwise_and(normal_image, normal_image, mask=thresholded_depth_map)

    # Save the separated images
    cv2.imwrite('results/separated_image_1.png', separated_image_1)
    cv2.imwrite('results/separated_image_2.png', separated_image_2)

# Example usage
depth_map_path = '3DGS_truck_depth/000001.png'
normal_image_path = '3DGS_truck/000001.png'
threshold = 89

separate_normal_image(depth_map_path, normal_image_path, threshold)
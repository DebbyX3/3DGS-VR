import cv2
import numpy as np
import os

def separate_normal_image_with_transparency(depth_map_path, normal_image_path, threshold, original_filename):

    # Load the depth map image
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

    # Apply Laplace function to find borders in the depth map
    kernel_size = 3
    laplacian = cv2.Laplacian(depth_map, cv2.CV_8U, ksize=kernel_size) # important: keep the data type as 8-bit unsigned integer

    # ********* TEST
    laplacian = cv2.Laplacian(depth_map, cv2.CV_16S, ksize=kernel_size) #test with different data type
    

    # Convert the result to 8-bit unsigned integer
    #laplacian = np.uint8(laplacian) #wrong but i need it to test
    laplacian = cv2.Laplacian(depth_map, cv2.CV_8U, ksize=kernel_size) #right

    # ********* END TEST
    
    # Perform the bitwise OR operation between borders and the depth map
    depth_map_with_borders = cv2.bitwise_or(depth_map, laplacian)

    # Load the normal image and convert it to BGRA (to include alpha channel)
    normal_image = cv2.imread(normal_image_path)
    normal_image_bgra = cv2.cvtColor(normal_image, cv2.COLOR_BGR2BGRA)

    # Apply threshold to the depth map
    # pixel below the threshold will be 0, above will be 255
    #
    # The output image will be black on the background 
    # and white on the foregound, which is the object.
    # Shows in white the near objects
    
    optimal_threshold, foreground_mask = cv2.threshold(depth_map_with_borders, threshold, 255, cv2.THRESH_OTSU) 
    
    print(optimal_threshold)

    # **** TEST

    #now only take the optimal threshold and USE IT ON THE ORIGINAL DEPTH MAP
    _, foreground_mask = cv2.threshold(depth_map, optimal_threshold, 255, cv2.THRESH_BINARY)

    # **** END TEST
    

    #foreground_mask = cv2.adaptiveThreshold(depth_map,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2) #bleah

    # Invert the thresholded depth map
    #
    # The output image is inverted: 
    # black on the foregound, which is the object
    # and white on the background
    background_mask = cv2.bitwise_not(foreground_mask)

    # Create alpha channels for separated images
    alpha_channel_background = background_mask
    alpha_channel_foreground = foreground_mask

    # Replace the alpha channel in the normal image with the new alpha channels
    separated_img_only_background = normal_image_bgra.copy()
    separated_img_only_background[:, :, 3] = alpha_channel_background

    separated_img_only_foreground = normal_image_bgra.copy()
    separated_img_only_foreground[:, :, 3] = alpha_channel_foreground

    # Save the separated images with transparency
    cv2.imwrite('results/bgs/' + original_filename + '_background.png', separated_img_only_background)
    cv2.imwrite('results/objects/' + original_filename + '_foreground.png', separated_img_only_foreground)

    # Save the foreground mask image
    cv2.imwrite('results/foreground_masks/' + original_filename + '_foreground_mask.png', foreground_mask)




# ****** PLEASE READ!!!
# Normal images AND depth images
# SHOULD HAVE THE SAME FILENAME TO MATCH THEM!!!!!
# This program presumes this condition is met!!!!

depth_map_path = '3DGS_truck_depth'
normal_image_path = '3DGS_truck'
threshold = 89


normal_img_directory = os.fsencode(r'./' + normal_image_path)

for file in os.listdir(normal_img_directory):
    filename = os.fsdecode(file)

    depth_img_filename = depth_map_path + r'/' + filename
    normal_img_filename = normal_image_path + r'/' + filename

    print(filename)

    separate_normal_image_with_transparency(depth_img_filename, normal_img_filename, threshold, filename)

    print('-----------------------------------')
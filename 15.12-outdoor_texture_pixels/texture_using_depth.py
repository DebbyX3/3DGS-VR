import open3d as o3d
import numpy as np
from PIL import Image
import os
import matplotlib
matplotlib.use('TkAgg')
import pylab as plt
import math
from collections import defaultdict
from pathlib import Path
import numpy.polynomial.polynomial as poly
import scipy

'''
Fit a depth map on another using a polynomial function
    depth_to_scale = the depth_map to scale/fit
    depth_to_base_on = the depth map used as a reference, on which depth_to_scale will be scaled on
    poly_deg = degree on polynomial

Returns: polynomial coefficients, scaled depth map
'''
def scale_texture_poly (depth_to_scale, depth_to_base_on, poly_deg):

    # ---- Flatten
    depth_to_scale_flat = depth_to_scale.flatten()
    depth_to_base_on_flat = depth_to_base_on.flatten()

    # find zeros indexes in colmap map
    zero_indexes_depth_to_base_on = np.where(depth_to_base_on_flat == 0)[0]
    
    # remove these indexes from both maps values
    depth_to_scale_values_clean = np.delete(depth_to_scale_flat, zero_indexes_depth_to_base_on)
    depth_to_base_on_values_clean = np.delete(depth_to_base_on_flat, zero_indexes_depth_to_base_on)

    # call the fitting method (new)
    # x = depth to scale clean vals
    # y = colmap to base on clean vals
    # deg = poly_deg

    # NOTA: MEGLIO USARE poly.polyfit E NON poly.Polynomial.fit, perchè 'fit' 
    # sono scalati con un linear mapping (not sure what it means), quindi per avere i coeff nell'unscaled data domain
    # bisogna fare .convert():
    # poly.Polynomial.fit(x, y, 2).convert().coef
    # oppure dare una finestra vuota nel dominio:
    # poly.Polynomial.fit(x, y, 2, domain=[]).coef
    # oppure dare come finestra in max e min dei dati:
    # poly.Polynomial.fit(x, y, 2, window=(x.min(), x.max())).coef    

    # create a polynomial in the form of:
    # c0 + c1 * x + c2 * x^2 + c3 * x^3 ... + cn * x^n
    # where c0...cn are the coefficients in the same orders that output from poly.polyfit

    #test con pesi
    #weights = depth_to_base_on_values_clean / depth_to_base_on_values_clean.max()  # Normalizza i pesi
    #weights[weights == 0] = 1  # Evita divisioni per 0

    function_coefs = poly.polyfit(depth_to_scale_values_clean, depth_to_base_on_values_clean, poly_deg) 

    #function_coefs = poly.polyfit(depth_to_scale_values_clean, depth_to_base_on_values_clean, poly_deg, w = weights) 
    print(function_coefs)

    function_polynomial = poly.Polynomial(function_coefs)
    print(function_polynomial)                        

    # apply the fitted function to each value of the depth anything pixel depth map to create a scaled version
    # Applica il polinomio a tutta la depth map in una sola operazione
    scaled_depth_map = poly.polyval(depth_to_scale, function_coefs)
    
    plt.imshow(scaled_depth_map, cmap='gray', vmin=0, vmax=255)
    plt.title("Fitted Depth Map")
    plt.show()

    # Punti per la curva del polinomio
    x_fit = np.linspace(depth_to_scale_values_clean.min(), depth_to_scale_values_clean.max(), 500)  # Asse X per il polinomio
    y_fit = function_polynomial(x_fit)  # Valori corrispondenti del polinomio

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(depth_to_scale_values_clean, depth_to_base_on_values_clean, color='blue', s=1, alpha=0.5, label='Dati Originali')
    plt.plot(x_fit, y_fit, color='red', linewidth=2, label=f'Polinomio di grado 3')
    plt.xlabel("Depth Anything (DA) Values")
    plt.ylabel("Colmap Values")
    plt.title("Fitting Polinomiale tra DA e Colmap")
    plt.legend()
    plt.grid()
    plt.show()

    return function_coefs, scaled_depth_map

# from colmap codebase
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


# ************************** PATHS **************************
cameraTxt_path = '../datasets/colmap_reconstructions/water_bottle_gui_pinhole_1camera/sparse/cameras.txt'
imagesTxt_path = '../datasets/colmap_reconstructions/water_bottle_gui_pinhole_1camera/sparse/images.txt'
imgs_folder = "../datasets/colmap_reconstructions/water_bottle_gui_pinhole_1camera/images"
depth_map_colmap_folder = '../datasets/colmap_reconstructions/water_bottle_gui_pinhole_1camera/stereo/depth_maps'

cameraTxt_path = '../datasets/colmap_reconstructions/colmap_output_simple_radial/sparse/cameras.txt'
imagesTxt_path = '../datasets/colmap_reconstructions/colmap_output_simple_radial/sparse/images.txt'
imgs_folder = "../datasets/colmap_reconstructions/colmap_output_simple_radial/dense/images"
depth_map_colmap_folder = '../datasets/colmap_reconstructions/colmap_output_simple_radial/dense/stereo/depth_maps'

cameraTxt_path = '../datasets/colmap_reconstructions/cavignal-bench_pinhole_1camera/sparse/cameras.txt'
imagesTxt_path = '../datasets/colmap_reconstructions/cavignal-bench_pinhole_1camera/sparse/images.txt'
imgs_folder = "../datasets/colmap_reconstructions/cavignal-bench_pinhole_1camera/dense/images"
depth_map_colmap_folder = '../datasets/colmap_reconstructions/cavignal-bench_pinhole_1camera/dense/stereo/depth_maps'
depth_map_da_non_metric_folder = '../depth-anything-estimations/non-metric_depths/cavignal_bench'
depth_map_da_metric_folder = '../depth-anything-estimations/metric_depths/cavignal_bench'


cameraTxt_path = '../datasets/colmap_reconstructions/cavignal-fountain_pinhole_1camera/sparse/cameras.txt'
imagesTxt_path = '../datasets/colmap_reconstructions/cavignal-fountain_pinhole_1camera/sparse/images.txt'
imgs_folder = "../datasets/colmap_reconstructions/cavignal-fountain_pinhole_1camera/dense/images"
depth_map_colmap_folder = '../datasets/colmap_reconstructions/cavignal-fountain_pinhole_1camera/dense/stereo/depth_maps'
depth_map_da_non_metric_folder = '../depth-anything-estimations/non-metric_depths/cavignal_fountain'
depth_map_da_metric_folder = '../depth-anything-estimations/metric_depths/cavignal-fountain_pinhole_1camera'

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

            break    # We only need the first camera intrinsics (assume all cameras have the same intrinsics)  

# Create the camera intrinsic matrix
intrinsic_matrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]])

print("--- Camera: ", camera_info[0][1])
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

#intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy) #alternatively
intrinsics = o3d.camera.PinholeCameraIntrinsic(camera_width, camera_height, intrinsic_matrix)

# ************************** EXTRACT EXTRINSICS FROM IMAGES.TXT FILE **************************
# Extrinsic matrix:
# [r1.1, r1.2, r1.3, tx]
# [r2.1, r2.2, r2.3, ty]
# [r3.1, r3.2, r3.3, tz]
# [0,    0,    0,    1 ]

# COMMON
texture_width, texture_height = 2048, 1024

# LOOP INFO
count = 0
count_imgs = 0
cameras_info = []
cameras_extrinsics = []
# Read 1 image every 'skip'
# e.g. If I have 10 imgs and skip = 3, read images:
# 3, 6, 9
skip = 5 # if 1: do not skip imgs

print("---Reading images from dataset: \t", imgs_folder)
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

                        # ------------- DEPTH

                        # PLEASE NOTE: 
                        # depth anything v2 uses an 'inverse' depth: the closer to the camera the point, the 'whiter' (= higher value) the pixel.
                        # Since i don't think is a good measure, and to standardize the method with colmap, i'm going to INVERT the values
                        #
                        # SO THE STANDARD IS:
                        # The LOWER the pixel value is, the CLOSER the pixel is to the camera 

                        # Take the image file name
                        img_filename = single_camera_info[9]

                        # ---- Read colmap depth map
                        # Numpy array with relative values, where the LOWER the value, the CLOSER the pixel is to the camera
                        depth_map_colmap_filename = img_filename + '.geometric.bin' # get the filename of the depth map
                        depth_map_colmap_path = os.path.join(depth_map_colmap_folder, depth_map_colmap_filename)
                        
                        depth_colmap = read_array(depth_map_colmap_path)

                        max = np.max(depth_colmap)
                        min = np.min(depth_colmap)

                        #norm_depth_colmap = ((depth_colmap - min)/(max-min)) * 255

                        # View grayscale from 0 to 255 (test)
                        #plt.imshow(norm_depth_colmap, cmap='gray', vmin=0, vmax=255)
                        plt.figure()
                        plt.imshow(depth_colmap, cmap='gray')
                        plt.title("Colmap")
                        plt.axis('off')
                        plt.show(block=False)

                        # ---- Read depth anything v2 (da) depth map - NON metric version
                        # Grayscale image from 0 to 255, where, originally, the HIGHER the color value, the CLOSER the pixel is
                        depth_map_da_non_metric_filename = Path(img_filename).stem + ".png" # get the filename of the depth map (remove file extension)
                        depth_map_da_non_metric_path = os.path.join(depth_map_da_non_metric_folder, depth_map_da_non_metric_filename)

                        depth_da_non_metric = np.asarray(Image.open(depth_map_da_non_metric_path).convert('L')) # convert 'L': open as greyscale 

                        # Need to invert the map to match the colmap rerpresentation of lower vals = closer point 
                        inverted_depth_da_non_metric = np.invert(depth_da_non_metric)

                        # View grayscale from 0 to 255 (test)
                        plt.figure()
                        plt.imshow(inverted_depth_da_non_metric, cmap='gray', vmin=0, vmax=255)
                        plt.title("Depth Anything NON metric (inverted)")
                        plt.axis('off')
                        plt.show(block=False)

                        # ---- Read depth anything v2 (da) depth map - METRIC version
                        # Grayscale image from 0 to 255, where, originally, the LOWER the color value, the CLOSER the pixel is
                        # The image is just a representation of the values normalized. 
                        # Instead, we have to use the numpy matrix generated by the metric estimation
                        '''
                        depth_map_da_metric_filename = Path(img_filename).stem + "_raw_depth_meter.npy" # (remove file extension)
                        depth_map_da_metric_path = os.path.join(depth_map_da_metric_folder, depth_map_da_metric_filename)

                        depth_da_metric = np.load(depth_map_da_metric_path)

                        # View grayscale from 0 to 255 (test)
                        plt.figure()
                        plt.imshow(depth_da_metric, cmap='gray', vmin=0, vmax=255)
                        plt.title("Depth Anything METRIC")
                        plt.axis('off')
                        plt.show(block=False)
                        '''

                        # ----------- FIND A FUNCTION
                        '''
                        - The Depth Anything (DA) map is dense/complete, but it does not have scaled or 'real' (not metric) values
                        - The Colmap map has scaled and 'real' (not metric) values, but is not complete: lots of point have 0 as 
                          depth because the method is unable to estimate it

                        So, we want to perform a scale: scale the DA map in the reference system of the colmap map using 
                        a function, that we try to fit as a polynomial on the colmap map. We should exclude the '0' points that
                        are not useful in the colmap map, and, similarly, exclude the same points in the DA map to avoid creating
                        a wrong fitted function.
                        At the end, we should apply the final function to the DA map and obtain a complete and scaled depth map.
                        Please note that we may need to find a function different FOR EACH image, since the DA maps are not consistent 
                        when generated from independent frames.
                        However, we can try to generate the depths using the 'video' method that depth anyting provides. 
                        Maybe in this way we can avoid finding a function for each frame, and just have one fitted function
                        '''

                        # ------------------- TEST 1: con depth DA non metriche
                        #_, scaled_depth_map = scale_texture_poly(inverted_depth_da_non_metric, depth_colmap, 3)
                        
                        # ------------------- TEST 2: Con depth DA metriche
                        #_, scaled_depth_map = scale_texture_poly(depth_da_metric, depth_colmap, 3)

                        # ------------------- TEST 3: Applica filtro mediana su ogni regione
                        ''' 
                        Perchè lo facciamo?
                        La depth di colmap è rumorosa. Ha un sacco di outlier in giro e dei buchi. Quello che voglio fare con questo
                        metodo è cercare di riempire i buchi in modo sensato, senza danneggiare la mappa.
                        Spero di ottenere questo e in più:
                        - una mappa più smooth e che abbia senso
                        - una mappa che poi posso fittare rispetto a quella di DA
                        - una mappa con i buchi 'piccoli' coperti, mentre con i buchi grandi no
                            - i buchi grandi sono il cielo e cose lontane. Se dopo questo procedimento (che posso applicare anche 2 volte?)
                              rimangono grandi buchi, allora posso dire che questi sono il cielo o oggetti lontani? 
                              Posso quindi artificialmente metterli a un valore molto alto?
                    
                        procedimento
                        - Prendi depth di DA non metrica, quindi con valori da 0 a 255
                        - Individua dei bucket di valori in questa mappa. Tipo, crea dei bucket ogni 10 valori, quindi 0-10, 10-20 etc...
                            - Magari, più avanti, si può fare un'analisi dei valori/dell'istogramma dei valori per fare dei bucket più sensati. 
                              Per es, se ho l'img che è tutta chiara e poco scura, allora il bucket dei chiari è più grande di quello degli scuri
                        - Per ogni bucket, cerca nella depth di DA le coordinate corrispondenti. 
                            - Nota che se siamo nello stesso bucket, è perchè allora i punti dovrebbero avere depth simili!
                        - Prendi le stesse coordinate, e considera solo quelle lì nella mappa di colmap. Di fatto guardo il valore in colmap negli stessi punti 
                            - Faccio così perchè voglio trovare in colmap le aree di profondità circa simili che mi dice DA, 
                              visto che è molto bravo a fare una segmentazione per depth
                        - Quando sono passata ai punti colmap, che ricordo essere solo quelli della regione del bucket di DA, cerco la depth + frequente (moda)
                        - Crea un'immagine/matrice da 0 della stessa dimensione, e riempila con i valori della moda
                        - Prendi i punti della regione trovata in colmap, e sostituiscili in questa nuova img.
                          Alla fine l'idea è avere una immagine/matrice con lo sfondo di moda e la regione con i valori delle depth di colmap per quel bucket
                        - Passa su questa matrice/img un filtro mediana, al quale do una certa finestra, che potrebbe essere anche variabile
                            - In alternativa, per non sprecare tempo a fare il filtro su tutta l'img nuova, posso farlo solo sulla regione, ma devo gestire bene 
                              i casi con gli edge. Inoltre, devo probabilmente scrivere da 0 una cosa del genere, mentre filtro mediana lo trovo in qualche lib
                            - tipo uso sempre l'idea di fare come sfondo e applico il filtro solo sulla maschera
                        - In più, posso calcolare dev std nei valori della finestra e setto una threshold (tipo 20). 
                          Se la thresh è 20 o meno, allora non filtro in quella finestra e vado avanti
                            - Questo lo faccio per evitare di piallare completamente zone abbastanza 'grandi' che invece vorrei tenere
                        '''
                        
                        bucket_step = 10
                        min_bucket = 0
                        max_bucket = 0

                        h_img, w_img = depth_colmap.shape[:2]
                        median_filtered_depth_map = np.zeros((h_img, w_img))
                        
                        while max_bucket < 255:
                            # Compute new buckets each loop
                            min_bucket = max_bucket
                            max_bucket = max_bucket + bucket_step

                            # If the max exceedes the limit, make it the limit
                            if max_bucket > 255:
                                max_bucket = 255

                            print("Bucket range: ", min_bucket, max_bucket)

                            # Extract indexes of the corresponding values between the bucket range in the DA map
                            # This creates a 'mask'
                            da_masked_indexes = np.where((inverted_depth_da_non_metric >= min_bucket) & \
                                                           (inverted_depth_da_non_metric < max_bucket))

                            # Extract the values of the same indexes/coords in the colmap map
                            colmap_masked_values = depth_colmap[da_masked_indexes]

                            # Find the most frequent value (mode)
                            # varianti:
                            # 1- Trova moda e basta
                            # 2- Trova moda togliendo gli 0
                            # 3- Trova moda togliendo gli 0 e arrotondando a 1 o 2 cifre dec
                            # 4- Trova moda arrotondando a 1 o 2 cifre dec
                            # 5- Trova mediana?
                            # (s-commenta variante che uso)

                            # --- Variante 1 - Trova moda e basta
                            '''
                            # find unique values in array along with their counts
                            vals, counts = np.unique(colmap_masked_values, return_counts=True)

                            # find mode index
                            mode_index = np.argwhere(counts == np.max(counts))

                            # print list of modes
                            mode_values = vals[mode_index].flatten().tolist()

                            # find how often mode occurs
                            mode_frequency = np.max(counts)
                            '''
                            
                            # --- Variante 2 - Trova moda togliendo gli 0
                            '''
                            # remove zeros 
                            colmap_masked_values_zero_indexes = np.where(colmap_masked_values == 0)
                            colmap_masked_values_no_zeros = np.delete(colmap_masked_values, colmap_masked_values_zero_indexes)

                            # find unique values in array along with their counts
                            vals, counts = np.unique(colmap_masked_values_no_zeros, return_counts=True)

                            # find mode index
                            mode_index = np.argwhere(counts == np.max(counts))

                            # print list of modes
                            mode_values = vals[mode_index].flatten().tolist()

                            # find how often mode occurs
                            mode_frequency = np.max(counts)

                            print("mode: ", mode_values)
                            print("mode freq: ", mode_frequency)
                            '''

                            # --- Variante 3 - Trova moda togliendo gli 0 e arrotondando a 1 o 2 cifre dec
                            
                            # remove zeros 
                            colmap_masked_values_zero_indexes = np.where(colmap_masked_values == 0)
                            colmap_masked_values_no_zeros = np.delete(colmap_masked_values, colmap_masked_values_zero_indexes)

                            # arrotonda a 2 cifre dec
                            colmap_masked_values_no_zeros_round = np.round(colmap_masked_values_no_zeros, decimals = 2)

                            # find unique values in array along with their counts
                            vals, counts = np.unique(colmap_masked_values_no_zeros_round, return_counts=True)

                            # find mode index
                            mode_index = np.argwhere(counts == np.max(counts))

                            # print list of modes
                            mode_values = list(vals[mode_index].flatten())

                            # find how often mode occurs
                            mode_frequency = np.max(counts)

                            print("mode: ", mode_values)
                            print("mode freq: ", mode_frequency)
                            

                            # --- Variante 4 - Trova moda arrotondando a 1 o 2 cifre dec
                            '''
                            # arrotonda a 2 cifre dec
                            colmap_masked_values_round = np.round(colmap_masked_values, decimals = 2)

                            # find unique values in array along with their counts
                            vals, counts = np.unique(colmap_masked_values_round, return_counts=True)

                            # find mode index
                            mode_index = np.argwhere(counts == np.max(counts))

                            # print list of modes
                            mode_values = list(vals[mode_index].flatten())

                            # find how often mode occurs
                            mode_frequency = np.max(counts)

                            print("mode: ", mode_values)
                            print("mode freq: ", mode_frequency)
                            '''

                            # --- crea immagine con stessa dim di depth colmap e riempi con moda
                            h_img, w_img = depth_colmap.shape[:2]
                            img_to_filter = np.full((h_img, w_img), mode_values[0])
                            # fill img with colmap mask
                            img_to_filter[da_masked_indexes] = colmap_masked_values

                            '''
                            plt.figure()
                            plt.imshow(img_to_filter, cmap='viridis', vmin=0, vmax=255)
                            plt.title("Img to filter - mode as background + mask with colmap vals in bucket")
                            plt.axis('off')
                            plt.show()
                            '''

                            # --- passa filtro mediana su questa immagine
                            current_img_median_filtered = scipy.signal.medfilt2d(img_to_filter, kernel_size = 31)

                            '''
                            plt.figure()
                            plt.imshow(img_median_filtered, cmap='viridis', vmin=0, vmax=255)
                            plt.title("Img median filtered")
                            plt.axis('off')
                            plt.show()         
                            '''   

                            # --- prendi i valori corrispondenti alle coord della maschera nell'immagine filtrata e salvali nella depth map finale 
                            median_filtered_depth_map[da_masked_indexes] = current_img_median_filtered[da_masked_indexes]      

                        plt.figure()
                        plt.imshow(median_filtered_depth_map, cmap='viridis', vmin=0, vmax=255)
                        plt.title("Median filtered depth map")
                        plt.axis('off')
                        plt.show()  

                        _, fitted_depth_map = scale_texture_poly(inverted_depth_da_non_metric, median_filtered_depth_map, 3)

                        # --- Cambia nella mappa con mediana i valori = 0 con quelli usciti dal fitting
                        # cerca indici valori 0
                        median_filtered_depth_map_zero_indexes = np.where(median_filtered_depth_map == 0)
                        # sostituisci gli indici con i valori della fitted
                        final_depth_map = median_filtered_depth_map
                        final_depth_map[median_filtered_depth_map_zero_indexes] = fitted_depth_map[median_filtered_depth_map_zero_indexes]

                        plt.figure()
                        plt.imshow(final_depth_map, cmap='viridis', vmin=0, vmax=255)
                        plt.title("Final depth map")
                        plt.axis('off')
                        plt.show()  

                        # ----------- IMAGE

                        # Read the image
                        img_path = os.path.join(imgs_folder, img_filename)
                        img = np.asarray(Image.open(img_path))   






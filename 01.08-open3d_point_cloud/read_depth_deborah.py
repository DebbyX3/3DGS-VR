import numpy as np
import pylab as plt
import matplotlib
matplotlib.use('TkAgg')

#--------------------- READ DEPTH MAP

# --- read depth_anything depth map
#map = np.load("depth_maps_metric_depthAny/mie_ext/20240802_150747_raw_depth_meter.npy") 
#map = np.load("depth_maps_metric_depthAny/mie_int/20240731_174758_raw_depth_meter.npy") 
#map = np.load("depth_maps_metric_depthAny/3DGS_truck/000001_raw_depth_meter.npy")

# --- read colmap depth map
map = np.load("../colmap_reconstructions/colmap_output_simple_radial/dense/stereo/depth_maps_npy_colmap/depth_map000001.png.geometric.bin.npy")

#--------------------- PRINT IMAGE INFORMATION
print("------ Image information")

# --- Size
print("Image dimensions:" , map.shape)

# --- Num of dimension
print("Image array has", map.ndim, "dimension(s)")

print("------ Min values")

# --- Min val
print("Min val (exclude 0):", np.min(map[np.nonzero(map)]))

# --- Indexes of min val
'''
Testati e non vanno bene perchè con le mappe di colmap crepano
#min_indexes = np.unravel_index((map[np.nonzero(map)]).argmin(), map.shape)
#oppure
#min_indexes = np.unravel_index((map[map>0]).argmin(), map.shape)
'''

# Find first occurrence of the min value and take the indexes
min_indexes = np.argwhere(map==np.min(map[np.nonzero(map)]))[0]

# asse 0 è sulla y, asse 1 è sulla x, quindi per quello sono scambiati
print("Indexes of min val (y,x):", min_indexes) 

# alternativa per trovare tutti gli indici del minimo, se ce ne fossero di più uguali: 
#min_indexes  = np.where(map == np.min(map[np.nonzero(map)]))

#--------------------- SHOW IMAGE IN PLOT

plt.figure()
plt.imshow(map)
# asse 0 è sulla y, asse 1 è sulla x
plt.scatter(x = min_indexes[1], y = min_indexes[0], color = 'red') # punto rosso nel punto val minore
plt.show()

#--------------------- MEAN OF VALUES OF BOTH DEPTH MAPS


mapMetric = np.load("../depth_maps_metric_depthAny/3DGS_truck/000001_raw_depth_meter.npy")
mapColmap = np.load("../colmap_reconstructions/colmap_output_simple_radial/dense/stereo/depth_maps_npy_colmap/depth_map000001.png.geometric.bin.npy")

averageMetric = mapMetric[np.nonzero(mapMetric)].mean()
averageColmap = mapColmap[np.nonzero(mapColmap)].mean()

correctionFactor = abs(averageMetric - averageColmap)

print("Avg metric map:", averageMetric)
print("Avg colmap map:", averageColmap)
print("Correction factor:", correctionFactor)

medianMetric = np.median(mapMetric[np.nonzero(mapMetric)])
medianColmap = np.median(mapColmap[np.nonzero(mapColmap)])

correctionFactor = abs(medianMetric - medianColmap)

print("Median metric map:", medianMetric)
print("Median colmap map:", medianColmap)
print("Correction factor:", correctionFactor)
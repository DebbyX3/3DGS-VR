from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import numpy as np

ply_input = "C:/Users/User/Desktop/Gaussian Splatting/gaussian-splatting-code/gaussian-splatting-deborah/output/smlPrk_fixPosScaleRot_2D_opac1_noDens_initColor_mask10_radius22-09_0-0329_2621442pt_7000_logScale/point_cloud/iteration_7000/point_cloud_origin.ply"
ply_output = "C:/Users/User/Desktop/Gaussian Splatting/gaussian-splatting-code/gaussian-splatting-deborah/output/smlPrk_fixPosScaleRot_2D_opac1_noDens_initColor_mask10_radius22-09_0-0329_2621442pt_7000_logScale/point_cloud/iteration_7000/pc-min-scale.ply"

# Carica il file .ply
plydata = PlyData.read(ply_input)

# ------------- DIVIDE SCALES ----------------

campi_da_dividere = ['scale_0', 'scale_1', 'scale_2']

multiply_by = 1.3

# Ottieni i dati dei vertici
vertices = plydata['vertex'].data

vertex_array = vertices.copy()

print(vertex_array['scale_0'][0])


# ------------- CLAMP VALUES TO A CERTAIN THRESHOLD ----------------


# === Dividi i campi
for campo in campi_da_dividere:
    if campo in vertex_array.dtype.names:
        print(f"Divido '{campo}' per {multiply_by}")
        vertex_array[campo] = vertex_array[campo] * multiply_by
    else:
        print(f"Campo '{campo}' non trovato, salto.")


# === Ricostruzione e salvataggio ===
new_vertex = PlyElement.describe(vertex_array, 'vertex')

# Mantieni altri elementi (es. 'face')
altri_elementi = [e for e in plydata.elements if e.name != 'vertex']

# Scrivi in formato binario
PlyData([new_vertex] + altri_elementi, text=False).write(ply_output)
print(f"\nFile salvato come '{ply_output}'")
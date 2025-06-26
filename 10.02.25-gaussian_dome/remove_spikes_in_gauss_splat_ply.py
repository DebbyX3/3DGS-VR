from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import numpy as np

ply_input = "C:/Users/User/Desktop/Gaussian Splatting/gaussian-splatting-code/gaussian-splatting-deborah/output/RemoveSpikes-FromTensboardNoBadImgs/point_cloud/iteration_7000/point_cloud_original.ply"
ply_output = "C:/Users/User/Desktop/Gaussian Splatting/gaussian-splatting-code/gaussian-splatting-deborah/output/RemoveSpikes-FromTensboardNoBadImgs/point_cloud/iteration_7000/point_cloud-NoSpikes.ply"

# Carica il file .ply
plydata = PlyData.read(ply_input)

# ------------- FIND MAX AND MIN VALUES IN SCALE 0, 1, 2 ----------------

# Scegli il nome del campo
scale0 = 'scale_0'
scale1 = 'scale_1'
scale2 = 'scale_2'

# Ottieni i dati dei vertici
vertices = plydata['vertex'].data

# Estrai i valori del campo desiderato
values_scale0 = vertices[scale0]
values_scale1 = vertices[scale1]
values_scale2 = vertices[scale2]

# Calcola minimo e massimo
min_scale0 = min(values_scale0)
max_scale0 = max(values_scale0)

min_scale1 = min(values_scale1)
max_scale1 = max(values_scale1)

min_scale2 = min(values_scale2)
max_scale2 = max(values_scale2)

# Stampa o usa i risultati
print(f"Valore minimo di {scale0}: {min_scale0}")
print(f"Valore massimo di {scale0}: {max_scale0}\n")

print(f"Valore minimo di {scale1}: {min_scale1}")
print(f"Valore massimo di {scale1}: {max_scale1}\n")

print(f"Valore minimo di {scale2}: {min_scale2}")
print(f"Valore massimo di {scale2}: {max_scale2}\n")


# **** grafico

campi = ['scale_0', 'scale_1', 'scale_2']
valori = {campo: vertices[campo] for campo in campi if campo in vertices.dtype.names}

# === Plot ===
plt.figure(figsize=(12, 4))
bins = 100  # Numero di bucket

for i, campo in enumerate(valori, start=1):
    plt.subplot(1, 3, i)
    plt.hist(valori[campo], bins=bins, color='skyblue', edgecolor='black')
    plt.title(f'Distribuzione di {campo}')
    plt.xlabel('Valore')
    plt.ylabel('Frequenza')
    plt.grid(True)

plt.tight_layout()
plt.show()

# ------------- CLAMP VALUES TO A CERTAIN THRESHOLD ----------------

# threshold
clamp_config = {
    'scale_0': (-4.5, -2),               # min, max
    'scale_1': (-4.5, -2),
    'scale_2': (-4.5, -2)
}

min_threshold = -1
vertices_copy = np.array(vertices)

# Applica il clamping
vertices_array = np.array(vertices)

for prop, (vmin, vmax) in clamp_config.items():
    if prop in vertices_array.dtype.names:
        original_values = vertices_array[prop]
        clamped_values = np.clip(original_values, vmin, vmax)
        vertices_array[prop] = clamped_values
        print(f"Clamped '{prop}': min={vmin}, max={vmax}")
    else:
        print(f"Proprietà '{prop}' non trovata nel file. Saltata.")


# === Ricostruzione e salvataggio ===
new_vertex_element = PlyElement.describe(vertices_array, 'vertex')

# Mantieni altri elementi (es. 'face')
altri_elementi = [e for e in plydata.elements if e.name != 'vertex']

# Scrivi in formato binario
#PlyData([new_vertex_element] + altri_elementi, text=False).write(ply_output)

print(f"\nFile salvato come '{ply_output}'")
import numpy as np
from plyfile import PlyData, PlyElement

#plydata = PlyData.read('C:/Users/User/Desktop/Gaussian Splatting/gaussian-splatting-new/gaussian-splatting/output/geodesic_fixPos_fixScaleMaxRad_fixRot_opac1_2dgau_noDens_exOpt_radius11-05_0-066_163842pts_7000/point_cloud/iteration_7000/point_cloud.ply')
plydata = PlyData.read('C:/Users/User/Desktop/Gaussian Splatting/gaussian-splatting-new/gaussian-splatting/output/geodesic_fixPos_fixScaleMaxRad_fixRot_opac1_2dgau_noDens_exOpt_radius11-05_0-066_163842pts_7000/input.ply ')

with open('./input.txt', 'w') as f:
    for element in plydata.elements:
        f.write(f"element {element.name} {len(element.data)}\n")
        for prop in element.properties:
            f.write(f"property {prop.name}\n")
        for item in element.data:
            f.write(" ".join(map(str, item)) + "\n")
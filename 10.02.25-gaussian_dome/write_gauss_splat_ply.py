import numpy as np
from plyfile import PlyData, PlyElement

# read a gaussian splatting txt file and save it as a ply file

def read_txt_to_numpy(filename):
    """
    Read a TXT file and convert it to a numpy array.
    
    Parameters:
    - filename: The path to the TXT file.
    
    Returns:
    - data: A numpy array containing the data from the TXT file.
    """
    data = np.loadtxt(filename, skiprows=63)  # Skip the header rows (first 63 lines)
    return data

def save_numpy_to_ply(data, filename):
    """
    Save a numpy array as a binary PLY file.
    
    Parameters:
    - data: A numpy array containing the data to save.
    - filename: The path to the PLY file.
    """
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('f_rest_0', 'f4'), ('f_rest_1', 'f4'), ('f_rest_2', 'f4'),
        ('f_rest_3', 'f4'), ('f_rest_4', 'f4'), ('f_rest_5', 'f4'),
        ('f_rest_6', 'f4'), ('f_rest_7', 'f4'), ('f_rest_8', 'f4'),
        ('f_rest_9', 'f4'), ('f_rest_10', 'f4'), ('f_rest_11', 'f4'),
        ('f_rest_12', 'f4'), ('f_rest_13', 'f4'), ('f_rest_14', 'f4'),
        ('f_rest_15', 'f4'), ('f_rest_16', 'f4'), ('f_rest_17', 'f4'),
        ('f_rest_18', 'f4'), ('f_rest_19', 'f4'), ('f_rest_20', 'f4'),
        ('f_rest_21', 'f4'), ('f_rest_22', 'f4'), ('f_rest_23', 'f4'),
        ('f_rest_24', 'f4'), ('f_rest_25', 'f4'), ('f_rest_26', 'f4'),
        ('f_rest_27', 'f4'), ('f_rest_28', 'f4'), ('f_rest_29', 'f4'),
        ('f_rest_30', 'f4'), ('f_rest_31', 'f4'), ('f_rest_32', 'f4'),
        ('f_rest_33', 'f4'), ('f_rest_34', 'f4'), ('f_rest_35', 'f4'),
        ('f_rest_36', 'f4'), ('f_rest_37', 'f4'), ('f_rest_38', 'f4'),
        ('f_rest_39', 'f4'), ('f_rest_40', 'f4'), ('f_rest_41', 'f4'),
        ('f_rest_42', 'f4'), ('f_rest_43', 'f4'), ('f_rest_44', 'f4'),
        ('opacity', 'f4'), ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]
    
    vertex_data = np.array([tuple(row) for row in data], dtype=dtype)
    ply_element = PlyElement.describe(vertex_data, 'vertex')
    PlyData([ply_element]).write(filename)

# Example usage
txt_filename = './input1replace.txt'
ply_filename = './input1replace.ply'

data = read_txt_to_numpy(txt_filename)
save_numpy_to_ply(data, ply_filename)


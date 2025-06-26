import math

def compute_fov(fx, fy, width, height):
    fov_x = 2 * math.atan(width / (2 * fx)) * 180 / math.pi
    fov_y = 2 * math.atan(height / (2 * fy)) * 180 / math.pi
    return fov_x, fov_y


# valori presi da una ricostruzione di colmap
# PINHOLE 1920 1080 1766.3281081713583 1729.1357296839633 960 540
fx = 1766.3281081713583
fy = 1729.1357296839633
width = 1920
height = 1080

fov_x, fov_y = compute_fov(fx, fy, width, height)

print(f"FOV X: {fov_x:.2f} degrees")
print(f"FOV Y: {fov_y:.2f} degrees")
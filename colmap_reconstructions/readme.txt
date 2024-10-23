To match colmap gui to GS colmap 'convert.py' script:

Feature extraction: 	camera PINHOLE (OPENCV and SIMPLE_PINHOLE might work as well)
			shared for all images TRUE (command line: --ImageReader.single.camera 1)

Feature matching: exhausting



"""issue""":
the official Gaussian splatting scripts sets the camera to OPENCV, but for some reason the output is always a PINHOLE camera??

Example cameras.txt:

# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
1 PINHOLE 1080 1920 1828.4992001220176 1854.4129318887899 540 960
HOW TO USE DEPTH ANYTHING

- ALWAYS activate env:
conda activate depth-anything-v2

- NON METRIC results
python run.py --encoder vitl --img-path "/home/labvips-ub/3D Gaussian Splatting/3DGS-VR/colmap_reconstructions/cavignal-bench_pinhole_1camera/dense/images/" --outdir "/home/labvips-ub/3D Gaussian Splatting/Depth-Anything-V2/outputs/" --pred-only --grayscale

- METRIC results

-- Indoor:
python run.py --encoder vitl --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth --max-depth 20 --img-path "C:\Users\User\Desktop\Gaussian Splatting\3DGS-VR\datasets\colmap_reconstructions\cavignal-fountain_pinhole_1camera\dense\images" --outdir "C:\Users\User\Desktop\Gaussian Splatting\3DGS-VR\depth-anything-estimations\metric_depths\cavignal-fountain_pinhole_1camera" --save-numpy --pred-only --grayscale


-- Outdoor:
python run.py --encoder vitl --load-from checkpoints/depth_anything_v2_metric_vkitti_vitl.pth --max-depth 80 --img-path "C:\Users\User\Desktop\Gaussian Splatting\3DGS-VR\datasets\colmap_reconstructions\cavignal-fountain_pinhole_1camera\dense\images" --outdir "C:\Users\User\Desktop\Gaussian Splatting\3DGS-VR\depth-anything-estimations\metric_depths\cavignal-fountain_pinhole_1camera" --save-numpy  --pred-only --grayscale

- VIDEO
python run_video.py --encoder vitl | vitg --video-path assets/examples_video --outdir video_depth_vis --pred-only --grayscale

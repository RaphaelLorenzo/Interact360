# Interact360

## Install
- Install `ffmpeg` if you dont have it already (for video extraction)
```
sudo apt-get install ffmpeg
```

- Create a conda environment and install dependencies (install torch accordingly to your CUDA version)
```
conda create --name interact360_env python=3.10
conda activate interact360_env
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Run

Place your video under `./demo/data` (or use `-v` to change the input video path and `-s` for the ouput saving path, then the appropriate `-d` to run outside of the directory)

From a video run these 3 commands to extract images, detect segment and track people inside, and assess if they are interacting 

<details>

<summary>Mask for interaction</summary>

By default we use the mask of `assets/default_mask_intersection_zone.png` update it according to your needs if you have a fixed interaction zone. If you have a moving interaction zone (like in SSUP) you can first track the interaction zone with :

```
python detect_segment_and_track.py -d ./demo/output/SSUP_SAMPLE_0/ -izt 2688,1042,4585,1919
```

Where `-izt 2688,1042,4585,1919` is the bounding box of the interaction zone in the first image then it will run a full pass of tracking to find the masks of the interaction zone. Then compute the convex hull of these masks. So you need a convex interaction zone, also this way of doing is not preferred because less reliable, a fix interaction zone - relatively to the camera - is better.


</details>


```
python ./process/extract_images.py -v ./demo/data/2025_06_13_11_15_31_Theta360.mkv
python ./process/detect_segment_and_track.py -d ./demo/output/2025_06_13_11_15_31_Theta360
python ./process/detect_interactions.py -d ./demo/output/2025_06_13_11_15_31_Theta360
```

Then you can see the result in the visualizer !
```
python visualize.py -d ./demo/output
```

## Acknoledgment 
SAM2
SAM2 Real Time : https://github.com/Gy920/segment-anything-2-real-time
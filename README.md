# Interact360

![cafet](./illustrations/cafet.gif)
![ssup](./illustrations/ssup.gif)

## Install
- Install `ffmpeg` if you dont have it already (for video extraction)
```
sudo apt-get install ffmpeg
```

- Create a conda environment and install dependencies
```
conda create --name interact360env python=3.10
conda activate interact360env
pip install -r requirements.txt
```

- Download the checkpoints (by default it will download necessary checkpoints for `SAM2.1-Tiny` and `Sapiens-03b`)
```
cd checkpoints
./download_ckpts.sh
cd ../
```

## Run

Place your video under `./demo/data` here we provide `SalleClubInteract-TC.mp4` (not a part of HUI360 or SSUP-A dataset)

```
python ./process/extract_images.py -i ./demo/data/SalleClubInteract-TC.mp4 -em time -met 60 -w 1
```

```
python ./process/detect_segment_and_track.py -d ./demo/output/SalleClubInteract -sm tiny -ybs 8 -sibs 8
```

```
python ./process/detect_interactions.py -d ./demo/output/SalleClubInteract
```

Then you can see the result in the visualizer !
```
python visualize.py -d ./demo/output
```




## Additional informations


<details>
<summary>Multiprocessing</summary>

In all scripts `-w` is used to specify the number of workers to use when things are parallelized, by default `-w` is `0` and scripts will use all cores, or half of the cores detected.

</details>




<details>
<summary>Optimized inference</summary>

The option `-vos` enable the use optimized inference with `torch.compile` for the image encoder in SAM2-1.

The option `-b16` enable the use of `bfloat16` checkpoints for Sapiens pose extraction (only available on some GPUs, and with versions of PyTorch that are compatible with the checkpoints)

</details>




<details>
<summary>Extracting video</summary>

**Episodes duration** : use `-em` to set the episode creation method (either `single` i.e. not episode, or `time` i.e. based on a maximum duration, `detection` to create episodes only when people are in front of the camera is not implemented here), use `-met` to set the maximum episode time in seconds

**Other options** : use `-h` to check them

</details>



<details>
<summary>Detection, segmentation and tracking</summary>

The process is explained in the publication. Many options are related to the filtering, and other filtering threhsolds are set as variable at the beggining of `detect_segment_and_track.py`

- `-ybs` refers to the batch size for YOLO initial detections, it will be multiplied by 4 since all images are subdivided in 4 overlapping regions
- `-sibs` refers to the batch size for SAM2-Image initial segmentation (it refers to the number of persons segmented simultaneously, but anyway the full images are processed one by one)
- In the tracking phase, tracks are processed one by one, and a single image at a time 

</details>




<details>
<summary>Mask for interaction</summary>

**Fixed interaction zone** : By default we use the mask of `assets/default_mask_intersection_zone.png` update it according to your needs if you have a fixed interaction zone. 

**Moving interaction zone** : If you have a moving interaction zone (like in SSUP) you can first track the interaction zone by passing its bouding box in the first image of your sequence as `-izt xmin,ymin,xmax,ymax`, note that `xmax` can be `> width` this is the way to track an object that appears on both sides of the equirectangular image. This will create a file `interact_zone_tracking_360.json` in each episode, that will be used by default by `detect_interaction.py`.
```
python detect_segment_and_track.py -d ./demo/output/SSUP_SAMPLE_0/ -izt 2688,1042,4585,1919
```

Where `-izt 2688,1042,4585,1919` is the bounding box of the interaction zone in the first image then it will run a full pass of tracking to find the masks of the interaction zone. Then compute the convex hull of these masks. So you need a convex interaction zone, also this way of doing is not preferred because less reliable, a fix interaction zone - relatively to the camera - is better.
</details>


## Acknoledgment 
SAM2
SAM2 Real Time : https://github.com/Gy920/segment-anything-2-real-time
ViTPose
Sapiens

